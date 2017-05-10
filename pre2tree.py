import sys

def tree(acts, sents):
	wid = 0
	re = ""
	for act in acts:
		if act[0] == 'S':
			re += " "+sents[wid]
			wid += 1
		elif act[0] == 'N':
			re += " ("+act[3:-1]
		else:
			re += ")"
	print re.strip()

if __name__ == "__main__":
	actions = []
	action = []	
	surfaces = []
	cnt = 0
	for line in open(sys.argv[1]):
		line = line.strip()
		if line == "":
			cnt = 0
			actions.append(action)
			action = []
			continue
		cnt += 1
		if cnt == 5:
			surfaces.append(line.split())
		if cnt >= 6:
			action.append(line)

	assert len(actions) == len(surfaces)

	for i in range(len(surfaces)):	
		tree(actions[i], surfaces[i]);
