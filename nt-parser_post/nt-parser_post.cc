#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "cnn/dict.h"
#include "cnn/cfsm-builder.h"

#include "nt-parser/oracle.h"
#include "nt-parser/pretrained.h"
#include "nt-parser/compressed-fstream.h"
#include "nt-parser/eval.h"

// dictionaries
cnn::Dict termdict, ntermdict, adict, posdict;
bool DEBUG;
volatile bool requested_stop = false;
unsigned IMPLICIT_REDUCE_AFTER_SHIFT = 0;
unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 60;
unsigned ACTION_DIM = 36;
unsigned PRETRAINED_DIM = 50;
unsigned LSTM_INPUT_DIM = 60;
unsigned POS_DIM = 10;

float ALPHA = 1.f;
unsigned N_SAMPLES = 1;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
float DROPOUT = 0.0f;
unsigned POS_SIZE = 0;
std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X
bool USE_POS = false;  // in discriminative parser, incorporate POS information in token embedding

using namespace cnn::expr;
using namespace cnn;
using namespace std;
namespace po = boost::program_options;


vector<unsigned> possible_actions;
unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training

void InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
        ("training_data,T", po::value<string>(), "List of Transitions - Training corpus")
        ("explicit_terminal_reduce,x", "[recommended] If set, the parser must explicitly process a REDUCE operation to complete a preterminal constituent")
        ("dev_data,d", po::value<string>(), "Development corpus")
        ("bracketing_dev_data,C", po::value<string>(), "Development bracketed corpus")

        ("test_data,p", po::value<string>(), "Test corpus")
        ("dropout,D", po::value<float>(), "Dropout rate")
        ("samples,s", po::value<unsigned>(), "Sample N trees for each test sentence instead of greedy max decoding")
        ("alpha,a", po::value<float>(), "Flatten (0 < alpha < 1) or sharpen (1 < alpha) sampling distribution")
        ("model,m", po::value<string>(), "Load saved model from this file")
        ("use_pos_tags,P", "make POS tags visible to parser")
        ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
        ("action_dim", po::value<unsigned>()->default_value(16), "action embedding size")
        ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
        ("input_dim", po::value<unsigned>()->default_value(32), "input embedding size")
        ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
        ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
        ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
        ("train,t", "Should training be run?")
        ("words,w", po::value<string>(), "Pretrained word embeddings")
        ("beam_size,b", po::value<unsigned>()->default_value(1), "beam size")
	("debug","debug")
        ("help,h", "Help");
  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
  if (conf->count("help")) {
    cerr << dcmdline_options << endl;
    exit(1);
  }
  if (conf->count("training_data") == 0) {
    cerr << "Please specify --traing_data (-T): this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.\n";
    exit(1);
  }
}

struct ParserBuilder {
  LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;
  LookupParameters* p_w; // word embeddings
  LookupParameters* p_t; // pretrained word embeddings (not updated)
  LookupParameters* p_nt; // nonterminal embeddings
  LookupParameters* p_a; // input action embeddings
  LookupParameters* p_pos; // pos embeddings (optional)
  Parameters* p_p2w;  // pos2word mapping (optional)
  Parameters* p_ptbias; // preterminal bias (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameters* p_ptW;    // preterminal W (used with IMPLICIT_REDUCE_AFTER_SHIFT)
  Parameters* p_pbias; // parser state bias
  Parameters* p_A; // action lstm to parser state
  Parameters* p_B; // buffer lstm to parser state
  Parameters* p_S; // stack lstm to parser state
  Parameters* p_w2l; // word to LSTM input
  Parameters* p_t2l; // pretrained word embeddings to LSTM input
  Parameters* p_ib; // LSTM input bias
  Parameters* p_cbias; // composition function bias
  Parameters* p_p2a;   // parser state to action
  Parameters* p_action_start;  // action bias
  Parameters* p_abias;  // action bias
  Parameters* p_buffer_guard;  // end of buffer
  Parameters* p_stack_guard;  // end of stack

  Parameters* p_b;
  Parameters* p_W_head;
  Parameters* p_W_dep;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      stack_lstm(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model),
      action_lstm(LAYERS, ACTION_DIM, HIDDEN_DIM, model),
      const_lstm_fwd(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      const_lstm_rev(LAYERS, LSTM_INPUT_DIM, LSTM_INPUT_DIM, model), // used to compose children of a node into a representation of the node
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_t(model->add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM})),
      p_nt(model->add_lookup_parameters(NT_SIZE, {LSTM_INPUT_DIM})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {ACTION_DIM})),
      p_pbias(model->add_parameters({HIDDEN_DIM})),
      p_A(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_B(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_S(model->add_parameters({HIDDEN_DIM, HIDDEN_DIM})),
      p_w2l(model->add_parameters({LSTM_INPUT_DIM, INPUT_DIM})),
      p_ib(model->add_parameters({LSTM_INPUT_DIM})),
      p_cbias(model->add_parameters({LSTM_INPUT_DIM})),
      p_p2a(model->add_parameters({ACTION_SIZE, HIDDEN_DIM})),
      p_action_start(model->add_parameters({ACTION_DIM})),
      p_abias(model->add_parameters({ACTION_SIZE})),

      p_buffer_guard(model->add_parameters({LSTM_INPUT_DIM})),
      p_stack_guard(model->add_parameters({LSTM_INPUT_DIM})),

      p_b(model->add_parameters({LSTM_INPUT_DIM})),
      p_W_head(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})),
      p_W_dep(model->add_parameters({LSTM_INPUT_DIM, LSTM_INPUT_DIM})){

    if (USE_POS) {
      p_pos = model->add_lookup_parameters(POS_SIZE, {POS_DIM});
      p_p2w = model->add_parameters({LSTM_INPUT_DIM, POS_DIM});
    }
    buffer_lstm = new LSTMBuilder(LAYERS, LSTM_INPUT_DIM, HIDDEN_DIM, model);
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {PRETRAINED_DIM});
      for (auto it : pretrained)
        p_t->Initialize(it.first, it.second);
      p_t2l = model->add_parameters({LSTM_INPUT_DIM, PRETRAINED_DIM});
    } else {
      p_t = nullptr;
      p_t2l = nullptr;
    }
  }

// checks to see if a proposed action is valid in discriminative models
static bool IsActionForbidden_Discriminative(const string& a, char prev_ac, char prev_ac_2, unsigned bsize, unsigned ssize, unsigned unary, bool temp) {
  bool is_shift = (a[0] == 'S');
  bool is_reduce = (a[0] == 'R');
  bool is_term = (a[0] == 'T');
  assert(is_shift || is_reduce || is_term) ;
  static const unsigned MAX_UNARY = 3;
//  if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;

  
  if (is_term){
    if(ssize != 2 || bsize != 1) return true;
  }

  if (is_shift){
    if(bsize == 1) return true;
    if(temp == true && prev_ac_2 == 'r') return true;
  }

  if (is_reduce){
    if(a[a.size()-3] == '*'){
      if(ssize == 3 && bsize == 1) return true;
    }
    if(a[a.size()-2] == 'r' && a[a.size()-3] == '*'){
      if(ssize <= 3) return true;
    }
    
    if(a[a.size()-2] == 's'){
      if(ssize <=1 || unary >= MAX_UNARY || temp == true) return true;
    }
    else{
      if(ssize <=2) return true;
    }
  }
  return false;

/*  if (IMPLICIT_REDUCE_AFTER_SHIFT) { //default 0
    // if a SHIFT has an implicit REDUCE, then only shift after an NT:
    if (is_shift && prev_a != 'N') return true;
  }
*/
  // be careful with top-level parens- you can only close them if you
  // have fully processed the buffer
/*  if (nopen_parens == 1 && bsize > 1) {
//    if (IMPLICIT_REDUCE_AFTER_SHIFT && is_shift) return true;
    if (is_reduce) return true;
  }
*/
  // you can't reduce after an NT action
//  if (is_reduce && prev_a == 'N') return true;
//  if (is_nt && bsize == 1) return true;
//  if (is_reduce && ssize < 3) return true;

  // TODO should we control the depth of the parse in some way? i.e., as long as there
  // are items in the buffer, we can do an NT operation, which could cause trouble
}


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// set sample=true to sample rather than max
vector<unsigned> log_prob_parser(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
                     bool is_evaluation,
                     bool sample = false) {
if(DEBUG) cerr << "sent size: " << sent.size()<<" total action: " << correct_actions.size()<<"\n";
    vector<unsigned> results;
    const bool build_training_graph = correct_actions.size() > 0;
    bool apply_dropout = (DROPOUT && !is_evaluation);
    stack_lstm.new_graph(*hg);
    action_lstm.new_graph(*hg);
    const_lstm_fwd.new_graph(*hg);
    const_lstm_rev.new_graph(*hg);
    stack_lstm.start_new_sequence();
    buffer_lstm->new_graph(*hg);
    buffer_lstm->start_new_sequence();
    action_lstm.start_new_sequence();
    if (apply_dropout) {
      stack_lstm.set_dropout(DROPOUT);
      action_lstm.set_dropout(DROPOUT);
      buffer_lstm->set_dropout(DROPOUT);
      const_lstm_fwd.set_dropout(DROPOUT);
      const_lstm_rev.set_dropout(DROPOUT);
    } else {
      stack_lstm.disable_dropout();
      action_lstm.disable_dropout();
      buffer_lstm->disable_dropout();
      const_lstm_fwd.disable_dropout();
      const_lstm_rev.disable_dropout();
    }
    // variables in the computation graph representing the parameters
    Expression pbias = parameter(*hg, p_pbias);
    Expression S = parameter(*hg, p_S);
    Expression B = parameter(*hg, p_B);
    Expression A = parameter(*hg, p_A);
    Expression ptbias, ptW;
    if (IMPLICIT_REDUCE_AFTER_SHIFT) {
      ptbias = parameter(*hg, p_ptbias);
      ptW = parameter(*hg, p_ptW);
    }
    Expression p2w;
    if (USE_POS) {
      p2w = parameter(*hg, p_p2w);
    }

    Expression ib = parameter(*hg, p_ib);
    Expression cbias = parameter(*hg, p_cbias);
    Expression w2l = parameter(*hg, p_w2l);
    Expression t2l;
    if (p_t2l)
      t2l = parameter(*hg, p_t2l);
    Expression p2a = parameter(*hg, p_p2a);
    Expression abias = parameter(*hg, p_abias);
    Expression action_start = parameter(*hg, p_action_start);

    Expression b_comp = parameter(*hg, p_b);
    Expression W_head = parameter(*hg, p_W_head);
    Expression W_dep  = parameter(*hg, p_W_dep);
    action_lstm.add_input(action_start);

    vector<Expression> buffer(sent.size() + 1);  // variables representing word embeddings
    // precompute buffer representation from left to right

    // in the discriminative model, here we set up the buffer contents
    for (unsigned i = 0; i < sent.size(); ++i) {
        int wordid = sent.raw[i]; // this will be equal to unk at dev/test
        if (build_training_graph && singletons.size() > wordid && singletons[wordid] && rand01() > 0.5)
          wordid = sent.unk[i];
if(DEBUG){
	cerr << termdict.Convert(wordid) <<" ";
}
        Expression w = lookup(*hg, p_w, wordid);

        vector<Expression> args = {ib, w2l, w}; // learn embeddings
        if (p_t && pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
          Expression t = const_lookup(*hg, p_t, sent.lc[i]);
          args.push_back(t2l);
          args.push_back(t);
        }
        if (USE_POS) {
          args.push_back(p2w);
          args.push_back(lookup(*hg, p_pos, sent.pos[i]));
        }
        buffer[sent.size() - i] = rectify(affine_transform(args));
    }
if(DEBUG){
	cerr<<"\n";
}
    // dummy symbol to represent the empty buffer
    buffer[0] = parameter(*hg, p_buffer_guard);
    for (auto& b : buffer)
      buffer_lstm->add_input(b);

    vector<Expression> stack;  // variables representing subtree embeddings
    stack.push_back(parameter(*hg, p_stack_guard));
    // drive dummy symbol on stack through LSTM
    stack_lstm.add_input(stack.back());
    
    vector<Expression> log_probs;
    unsigned action_count = 0;  // incremented at each prediction
    vector<unsigned> current_valid_actions;
    vector<unsigned> unary;
    vector<bool> temp_node;
    char prev_a = '0';
    char prev_a_2 = '0';
//    while(stack.size() > 2 || buffer.size() > 1) {
    while(true){
	if(prev_a == 'T') break;
      // get list of possible actions for the current parser state
if(DEBUG) cerr<< "action_count: " << action_count <<"\n";
      current_valid_actions.clear();
if(DEBUG) {
	cerr<<"unary: ";
	for(unsigned i = 0; i < unary.size(); i++){
		cerr<<unary[i]<<" ";
	}
	cerr<<"\n";
		
	cerr<<"temp_node: ";
	for(unsigned i = 0; i < temp_node.size(); i++){
                cerr<<temp_node[i]<<" ";
        }
        cerr<<"\n";

}
      for (auto a: possible_actions) {
        if (IsActionForbidden_Discriminative(adict.Convert(a), prev_a, prev_a_2, buffer.size(), stack.size(), (unary.size() == 0 ? 0 : unary.back()), (temp_node.size() == 0 ? false : temp_node.back())))
          continue;
        current_valid_actions.push_back(a);
      }
if(DEBUG){
	cerr <<"current_valid_actions: "<<current_valid_actions.size()<<" :";
	for(unsigned i = 0; i < current_valid_actions.size(); i ++){
		cerr<<adict.Convert(current_valid_actions[i])<<" ";
	}
	cerr <<"\n";

	if(build_training_graph){
	unsigned j = 999;
	for(unsigned i = 0; i < current_valid_actions.size(); i ++){
                if(current_valid_actions[i] == correct_actions[action_count]) {j = i; break;}
        }
	if(j == 999){
		cerr<<"gold out\n";
		exit(1);
	}
	}
	cerr<<"current_valud_actions ok\n";
}
      //cerr << "valid actions = " << current_valid_actions.size() << endl;

      // p_t = pbias + S * slstm + B * blstm + A * almst
      //
      Expression stack_summary = stack_lstm.back();
      Expression action_summary = action_lstm.back();
      Expression buffer_summary = buffer_lstm->back();
      if (apply_dropout) {
        stack_summary = dropout(stack_summary, DROPOUT);
        action_summary = dropout(action_summary, DROPOUT);
        buffer_summary = dropout(buffer_summary, DROPOUT);
      }

if(DEBUG){
	hg->incremental_forward();
        cerr<<"summary ok\n";
}

      Expression p_t = affine_transform({pbias, S, stack_summary, B, buffer_summary, A, action_summary});
      Expression nlp_t = rectify(p_t);
      //if (build_training_graph) nlp_t = dropout(nlp_t, 0.4);
      // r_t = abias + p2a * nlp
      Expression r_t = affine_transform({abias, p2a, nlp_t});
      if (sample && ALPHA != 1.0f) r_t = r_t * ALPHA;
      // adist = log_softmax(r_t, current_valid_actions)
      Expression adiste = log_softmax(r_t, current_valid_actions);
      vector<float> adist = as_vector(hg->incremental_forward());
      double best_score = adist[current_valid_actions[0]];
      unsigned model_action = current_valid_actions[0];
      if (sample) {
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(adist[current_valid_actions[w]]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        model_action = current_valid_actions[w];
      } else { // max
        for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
          if (adist[current_valid_actions[i]] > best_score) {
            best_score = adist[current_valid_actions[i]];
            model_action = current_valid_actions[i];
          }
        }
      }
      unsigned action = model_action;
      if (build_training_graph) {  // if we have reference actions (for training) use the reference action
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        if (model_action == action) { (*right)++; }
      } else {
        //cerr << "Chosen action: " << adict.Convert(action) << endl;
      }
      //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.Convert(i) << ':' << adist[i]; }
      //cerr << endl;
      ++action_count;
      log_probs.push_back(pick(adiste, action));
      results.push_back(action);
if(DEBUG){

      cerr << "MODEL_ACT: " << model_action <<" ";
      cerr << adict.Convert(model_action)<<" ";
      cerr <<"GOLD_ACT: " << action << " ";
      cerr  <<adict.Convert(action)<<"\n";
}
      // add current action to action LSTM
      Expression actione = lookup(*hg, p_a, action);
      action_lstm.add_input(actione);

if(DEBUG){
	cerr<<"actions lookup ok\n";
}
      // do action
      const string& actionString=adict.Convert(action);
      //cerr << "ACT: " << actionString << endl;
      const char ac = actionString[0];
      const char ac_2 = actionString[actionString.size()-2];
      const char ac_3 = actionString[actionString.size()-3];

//if(DEBUG){
//	cerr<<"ac: "<< ac << " ac_2: " << ac_2 <<" ac_3: " <<ac_3<<"\n";
//}
      if (ac =='S') {  // SHIFT
        assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)
        stack.push_back(buffer.back());
        stack_lstm.add_input(buffer.back());
        buffer.pop_back();
        buffer_lstm->rewind_one_step();
	unary.push_back(0);
	temp_node.push_back(false);
      } else if (ac == 'R'){ // REDUCE
        // find what paren we are closing
        // REDUCE(NP-l*)
        auto lb = actionString.find('(');
	auto rb = actionString.find('-');
	string nterm = actionString.substr(lb+1, rb-lb-1);
	int nterm_idx = ntermdict.Convert(nterm);
//if(DEBUG){
//	cerr<<"nonterminal: "<<nterm_idx<<" "<<nterm<<"\n";
//}

	const_lstm_fwd.start_new_sequence();
        const_lstm_rev.start_new_sequence();
	Expression nonterminal = lookup(*hg, p_nt, nterm_idx);

	Expression head;
	Expression dep;
	if(ac_2 == 's'){
		assert(stack.size() > 1);
		head = stack.back();
		stack_lstm.rewind_one_step();
		stack.pop_back();
		unary[unary.size()-1] += 1;
	}
	else if(ac_2 == 'r'){
		assert(stack.size() > 2);
		head = stack.back();
		stack_lstm.rewind_one_step();
		stack.pop_back();
		dep = stack.back();
		stack_lstm.rewind_one_step();
		stack.pop_back();
		unary.pop_back();
		unary.pop_back();
		unary.push_back(0);
		temp_node.pop_back();
		temp_node.pop_back();
		temp_node.push_back((ac_3 == '*'));
	}
	else if(ac_2 == 'l'){
		assert(stack.size() > 2);
		dep = stack.back();
                stack_lstm.rewind_one_step();
		stack.pop_back();
                head = stack.back();
                stack_lstm.rewind_one_step();
		stack.pop_back();
		unary.pop_back();
                unary.pop_back();
                unary.push_back(0);
		temp_node.pop_back();
                temp_node.pop_back();
                temp_node.push_back((ac_3 == '*'));
	}
	else{
		cerr<<"label error!\n";
		abort();
	}

	const_lstm_fwd.add_input(nonterminal);
        const_lstm_rev.add_input(nonterminal);
        
	const_lstm_fwd.add_input(head);
	if(ac_2 == 'l' || ac_2 == 'r')
        const_lstm_rev.add_input(dep);

	if(ac_2 == 'l' || ac_2 == 'r')
	const_lstm_fwd.add_input(dep);
        const_lstm_rev.add_input(head);
	
	Expression cfwd = const_lstm_fwd.back();
        Expression crev = const_lstm_rev.back();

	Expression comp = rectify(affine_transform({b_comp, W_head, cfwd, W_dep, crev})); 
        stack_lstm.add_input(comp);
        stack.push_back(comp);
      }
      else{// TERMINATE
      }
      prev_a = ac;
      prev_a_2 = ac_2;
    }
    if (build_training_graph && action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }
    assert(stack.size() == 2); // guard symbol, root
    assert(buffer.size() == 1); // guard symbol
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return results;
  }




struct ParserState {
  LSTMBuilder stack_lstm;
  LSTMBuilder *buffer_lstm;
  LSTMBuilder action_lstm;
  vector<Expression> buffer;
  vector<int> bufferi;
  LSTMBuilder const_lstm_fwd;
  LSTMBuilder const_lstm_rev;

  vector<Expression> stack;
  vector<int> stacki;
  vector<unsigned> results;  // sequence of predicted actions
  bool complete;
  vector<Expression> log_probs;
  double score;
  int action_count;
  int nopen_parens;
  char prev_a;
};


struct ParserStateCompare {
  bool operator()(const ParserState& a, const ParserState& b) const {
    return a.score > b.score;
  }
};

static void prune(vector<ParserState>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
  //cerr << "PRUNE\n";
  //for (unsigned i = 0; i < pq.size(); ++i) {
  //  cerr << pq[i].score << endl;
  //}
}

static bool all_complete(const vector<ParserState>& pq) {
  for (auto& ps : pq) if (!ps.complete) return false;
  return true;
}


};
void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv, 1989121011);

  cerr << "COMMAND LINE:"; 
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;

  po::variables_map conf;
  InitCommandLine(argc, argv, &conf);
  DEBUG = conf.count("debug");
  IMPLICIT_REDUCE_AFTER_SHIFT = conf.count("explicit_terminal_reduce") == 0;
  USE_POS = conf.count("use_pos_tags");
  if (conf.count("dropout"))
    DROPOUT = conf["dropout"].as<float>();
  LAYERS = conf["layers"].as<unsigned>();
  INPUT_DIM = conf["input_dim"].as<unsigned>();
  PRETRAINED_DIM = conf["pretrained_dim"].as<unsigned>();
  HIDDEN_DIM = conf["hidden_dim"].as<unsigned>();
  ACTION_DIM = conf["action_dim"].as<unsigned>();
  LSTM_INPUT_DIM = conf["lstm_input_dim"].as<unsigned>();
  POS_DIM = conf["pos_dim"].as<unsigned>();
  if (conf.count("train") && conf.count("dev_data") == 0) {
    cerr << "You specified --train but did not specify --dev_data FILE\n";
    return 1;
  }
  if (conf.count("alpha")) {
    ALPHA = conf["alpha"].as<float>();
    if (ALPHA <= 0.f) { cerr << "--alpha must be between 0 and +infty\n"; abort(); }
  }
  if (conf.count("samples")) {
    N_SAMPLES = conf["samples"].as<unsigned>();
    if (N_SAMPLES == 0) { cerr << "Please specify N>0 samples\n"; abort(); }
  }
  
  ostringstream os;
  os << "ntparse"
     << (USE_POS ? "_pos" : "")
     << '_' << IMPLICIT_REDUCE_AFTER_SHIFT
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << '_' << ACTION_DIM
     << '_' << LSTM_INPUT_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "PARAMETER FILE: " << fname << endl;
  bool softlinkCreated = false;

  Model model;

  parser::TopDownOracle corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::TopDownOracle dev_corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::TopDownOracle test_corpus(&termdict, &adict, &posdict, &ntermdict);
  corpus.load_oracle(conf["training_data"].as<string>(), true);	
  corpus.load_bdata(conf["bracketing_dev_data"].as<string>());

  if (conf.count("words"))
    parser::ReadEmbeddings_word2vec(conf["words"].as<string>(), &termdict, &pretrained);

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.Freeze();
  termdict.SetUnk("UNK"); // we don't actually expect to use this often
     // since the Oracles are required to be "pre-UNKified", but this prevents
     // problems with UNKifying the lowercased data which needs to be loaded
  adict.Freeze();
  ntermdict.Freeze();
  posdict.Freeze();

  {  // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  if (conf.count("dev_data")) {
    cerr << "Loading validation set\n";
    dev_corpus.load_oracle(conf["dev_data"].as<string>(), false);
  }
  if (conf.count("test_data")) {
    cerr << "Loading test set\n";
    test_corpus.load_oracle(conf["test_data"].as<string>(), false);
  }

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.Convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.Convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
  }

  NT_SIZE = ntermdict.size();
  POS_SIZE = posdict.size();
  VOCAB_SIZE = termdict.size();
  ACTION_SIZE = adict.size();
  possible_actions.resize(adict.size());
  for (unsigned i = 0; i < adict.size(); ++i)
    possible_actions[i] = i;

  ParserBuilder parser(&model, pretrained);
  if (conf.count("model")) {
    ifstream in(conf["model"].as<string>().c_str());
    boost::archive::text_iarchive ia(in);
    ia >> model;
  }
  cerr<<"Actions:\n";
  for(unsigned i = 0; i < adict.size(); ++i){
    cerr<<adict.Convert(i)<<"\n";
  }
  cerr<<"Postags:\n";
  for(unsigned i = 0; i < posdict.size(); ++i){
    cerr<<posdict.Convert(i)<<"\n";
  }
  cerr<<"Nonterm:\n";
  for(unsigned i = 0; i < ntermdict.size(); ++i){
    cerr<<ntermdict.Convert(i)<<"\n";
  }
  //TRAINING
  if (conf.count("train")) {
    signal(SIGINT, signal_callback_handler);
    SimpleSGDTrainer sgd(&model);
    //AdamTrainer sgd(&model);
    //MomentumSGDTrainer sgd(&model);
    //sgd.eta_decay = 0.08;
    sgd.eta_decay = 0.05;
    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min((int)status_every_i_iterations, (int)corpus.sents.size());
    unsigned si = corpus.sents.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sents.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev_err = 9e99;
    double bestf1=0.0;
    vector<string> model_path;
    //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.sents.size()) {
             si = 0;
             if (first) { first = false; } else { sgd.update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<int>& actions=corpus.actions[order[si]];
           ComputationGraph hg;
           parser.log_prob_parser(&hg,sentence,actions,&right,false);
           double lp = as_scalar(hg.incremental_forward());
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward();
           sgd.update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
           words += sentence.size();
      }
      sgd.status();
      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<
         /*" |time=" << put_time(localtime(&time_now), "%c %Z") << ")\tllh: "<< */
        ") per-action-ppl: " << exp(llh / trs) << " per-input-ppl: " << exp(llh / words) << " per-sent-ppl: " << exp(llh / status_every_i_iterations) << " err: " << (trs - right) / trs << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << endl;
      llh = trs = right = words = 0;

      static int logc = 0;
      ++logc;
      if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = dev_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
	ostringstream os;
        os << "parser_dev_eval." << getpid() << ".txt";
        const string pfx = os.str();
        ofstream out(pfx.c_str());
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<int>& actions=dev_corpus.actions[sii];
	   dwords += sentence.size();
           {  ComputationGraph hg;
              parser.log_prob_parser(&hg,sentence,actions,&right,true);
              double lp = as_scalar(hg.incremental_forward());
              llh += lp;
           }
           ComputationGraph hg;
           vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,vector<int>(),&right,true);
           int ti = 0;
           for (auto a : pred) {
		out << adict.Convert(a) << endl;
           }
           out << endl;
           double lp = 0;
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;

	std::string command_2="python post2tree.py "+pfx+ " " + conf["dev_data"].as<string>() + " > "+pfx+".binary" ;
	const char* cmd_2=command_2.c_str();
	cerr<<system(cmd_2)<<"\n";

	std::string command_1="python unbinarize.py "+ pfx+".binary" +" > "+pfx+".eval" ;
        const char* cmd_1=command_1.c_str();
        cerr<<system(cmd_1)<<"\n";

        //parser::EvalBResults res = parser::Evaluate("foo", pfx);
	std::string command="python remove_dev_unk.py "+ corpus.devdata + " "+pfx+".eval "+" > evaluable.txt";
	const char* cmd=command.c_str();
	system(cmd);

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" evaluable.txt > evalbout.txt";
        const char* cmd2=command2.c_str();

        system(cmd2);
        
        std::ifstream evalfile("evalbout.txt");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        bool found=0;
        while (getline(evalfile, lineS) && !newfmeasure){
		if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
			//std::cout<<lineS<<"\n";
			strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;     // alias of size_t

		        newfmeasure = std::stod (strfmeasure,&sz);
			//std::cout<<strfmeasure<<"\n";
		}
        }
        
 
        
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\tllh=" << llh << " ppl: " << exp(llh / dwords) << " f1: " << newfmeasure << " err: " << err << "\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << endl;
//        if (err < best_dev_err && (tot_seen / corpus.size()) > 1.0) {
       if (newfmeasure>bestf1) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_err = err;
	  bestf1=newfmeasure;
	  ostringstream part_os;
  	  part_os << "ntparse"
     	      << (USE_POS ? "_pos" : "")
              << '_' << IMPLICIT_REDUCE_AFTER_SHIFT
              << '_' << LAYERS
              << '_' << INPUT_DIM
              << '_' << HIDDEN_DIM
              << '_' << ACTION_DIM
              << '_' << LSTM_INPUT_DIM
              << "-pid" << getpid() 
	      << "-part" << (tot_seen/corpus.size()) << ".params";
 	  
	  const string part = part_os.str();
 
          ofstream out("model/"+part);
          boost::archive::text_oarchive oa(out);
          oa << model;
	  system((string("cp ") + pfx + string(" ") + pfx + string(".best")).c_str());
	  if(model_path.size() == 5){
                const string p = model_path[0];
                system((string("rm ")+p).c_str());
                for(unsigned i = 0; i < 4; i ++){
                        model_path[i] = model_path[i+1];
                }
          }
	  model_path.push_back("model/"+part);
          // Create a soft link to the most recent model in order to make it
          // easier to refer to it in a shell script.
          /*if (!softlinkCreated) {
            string softlink = " latest_model";
            if (system((string("rm -f ") + softlink).c_str()) == 0 && 
                system((string("ln -s ") + fname + softlink).c_str()) == 0) {
              cerr << "Created " << softlink << " as a soft link to " << fname 
                   << " for convenience." << endl;
            }
            softlinkCreated = true;
          }*/
        }
      }
    }
  } // should do training?
  if (test_corpus.size() > 0) { // do test evaluation
        bool sample = conf.count("samples") > 0;
        unsigned test_size = test_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        auto t_start = chrono::high_resolution_clock::now();
	const vector<int> actions;
        /*for (unsigned sii = 0; sii < test_size; ++sii) {
           const auto& sentence=test_corpus.sents[sii];
           dwords += sentence.size();
           for (unsigned z = 0; z < N_SAMPLES; ++z) {
             ComputationGraph hg;
             vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,actions,&right,sample,true);
             double lp = as_scalar(hg.incremental_forward());
             cout << sii << " ||| " << -lp << " |||\n";
             int ti = 0;
             for (auto a : pred) {
             	cout << adict.Convert(a);
		if (adict.Convert(a) == "SHIFT"){
			cout<<" "<<termdict.Convert(sentence.raw[ti++]);
		}
		cout << endl;
	     }
             cout << endl;
           }
       }*/
        ofstream out("test.act");
        t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < test_size; ++sii) {
           const auto& sentence=test_corpus.sents[sii];
           const vector<int>& actions=test_corpus.actions[sii];
           dwords += sentence.size();
           {  ComputationGraph hg;
              parser.log_prob_parser(&hg,sentence,actions,&right,true);
              double lp = as_scalar(hg.incremental_forward());
              llh += lp;
           }
           ComputationGraph hg;
           vector<unsigned> pred = parser.log_prob_parser(&hg,sentence,vector<int>(),&right,true);
           int ti = 0;
           for (auto a : pred) {
           	out << adict.Convert(a) << endl;
	   }
           out << endl;
           double lp = 0;
           trs += actions.size();
        }
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;

        /*std::string command_1="python mid2tree.py test.act " + conf["test_data"].as<string>() + " > test.eval" ;
        const char* cmd_1=command_1.c_str();
        cerr<<system(cmd_1)<<"\n";

	//parser::EvalBResults res = parser::Evaluate("foo", pfx);
        std::string command="python remove_dev_unk.py "+ corpus.devdata +" test.eval > evaluable.txt";
        const char* cmd=command.c_str();
        system(cmd);

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" evaluable.txt > evalbout.txt";
        const char* cmd2=command2.c_str();

        system(cmd2);

        std::ifstream evalfile("evalbout.txt");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        bool found=0;
        while (getline(evalfile, lineS) && !newfmeasure){
                if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
                        //std::cout<<lineS<<"\n";
                        strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;
                        newfmeasure = std::stod (strfmeasure,&sz);
                        //std::cout<<strfmeasure<<"\n";
                }
        }

       cerr<<"F1score: "<<newfmeasure<<"\n";
    */
  }
}
