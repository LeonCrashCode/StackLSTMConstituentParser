import sys

def tree(acts, sents):
	re = []
	wid = 0
	for act in acts:
                if act[0] == 'S':
                        re.append("( XX t "+sents[wid]+" )")
                        wid += 1
                elif act[0] == 'R':
                        label,aux = act[7:-1].split('-')
                        if aux == 's':
                                re[-1] = "( "+label+" s "+re[-1]+" )"
                        else:
                                top1 = re[-1]
                                top2 = re[-2]
                                re = re[:-2]
                                aux = "".join(list(aux)[::-1])
                                re.append("( "+label+" "+aux+" "+top2+" "+top1+" )")
                else:
                        assert len(re) == 1
	print re[0]

if __name__ == "__main__":
	actions = []
	action = []	
	for line in open(sys.argv[1]):
		line = line.strip()
		if line == "":
			actions.append(action[:-1])
			action = []
		else:
			action.append(line)

	surfaces = []
	cnt = 0
	for line in open(sys.argv[2]):
		line = line.strip()
		cnt += 1
		if cnt == 3:
			surfaces.append(line.split())
		if line == "":
			cnt = 0

	assert len(actions) == len(surfaces)

	for i in range(len(surfaces)):	
		tree(actions[i], surfaces[i]);
