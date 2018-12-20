
def clear(word):
	while len(word) > 1:
		if word[0].isalnum() and word[-1].isalnum(): break
		if not word[0].isalnum(): word = word[1:]
		if not word[-1].isalnum(): word = word[:-1]
	
	# something else
	
	return word


def make_set(filename):
	s = set()
	with open("data/" + str(filename) + ".txt", 'r', encoding='utf8') as article:
		for line in article:
			for word in line.strip().split():
				word = clear(word)
				s.update({word.lower()})
	return s

def indexation(noa):
	print("\n--- INDEXATOR ---")
	all = set()
	part = set()
	index = dict()

	for i in range(1, noa + 1):# 									make set of all words
		all.update(make_set(i))


	print("--- indexation ---")
	for i in range(1, noa + 1):#									indexation
		print("--- file №" + str(i) + " ---")
		part = make_set(i)
		for word in all:
			if word in part:
				if word not in index.keys():
					index.update({word : [i]})
				else:
					index[word] += [i]
	return index

