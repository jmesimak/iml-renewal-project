import pprint
from collections import Counter, OrderedDict
from math import ceil, log
from sklearn.datasets import fetch_20newsgroups
import numpy as np

pp = pprint.PrettyPrinter(indent=4)

def load_data():
	subset = 'train'
	news20 = fetch_20newsgroups(subset=subset)
	label_names = news20.target_names
	labels = np.array(news20.target)
	posts = np.array(news20.data)
	sample_size = len(posts)
	print "Taking a sample of size {} from the 20newsgroups subset: {}".format(sample_size, subset)
	idx = np.random.choice(np.arange(len(posts)), sample_size, replace=False)
	posts = posts[idx]
	labels = labels[idx]
	return posts, labels, label_names

def construct_sets(newsgroup_posts, newsgroup_labels):
	organized = {}
	for idx, post in enumerate(newsgroup_posts):
		label = newsgroup_labels[idx]
		if label not in organized:
			organized[label] = []

		organized[label].append(post)

	train = []
	test = []

	for label in organized.keys():
		label_posts = organized[label]
		post_training = [(lpost, label) for lpost in label_posts[:int(ceil((len(label_posts)+0.01)*0.9))]]
		post_test = [(lpost, label) for lpost in label_posts[int(len(label_posts)*0.9)+1:]]
		assert(len(label_posts) == (len(post_training) + len(post_test)))
		train.extend(post_training)
		test.extend(post_test)

	print "Constructed training set of size {} and test set of size {}".format(len(train), len(test))
	return train, test, organized

def calculate_word_occurrences(training_set):
	print "Calculating word occurrences for training set which contains {} items".format(len(training_set))
	occurrences = {}
	all_occurrences = Counter()
	for post, label in training_set:
		counted = Counter(post.split())
		if label not in occurrences:
			occurrences[label] = counted
		else:
			occurrences[label] += counted
		all_occurrences += counted

	print "Applying Laplace smoothing to {} distinct words".format(len(all_occurrences.keys()))
	print "which has {} words in total".format(sum(all_occurrences.values()))
	print "This means that the average post has {} words".format(sum(all_occurrences.values()) / len(training_set))
	laplace_smoothing_constant = 0.0000001
	for word in all_occurrences:

		for label in occurrences:
			if word not in occurrences[label]:
				occurrences[label][word] = laplace_smoothing_constant
			else:
				occurrences[label][word] += laplace_smoothing_constant

		all_occurrences[word] += (len(occurrences) * laplace_smoothing_constant)

	print "After Laplace smoothing we consider the total amount of words: {}".format(sum(all_occurrences.values()))
	return all_occurrences, occurrences

def classify(post, word_occurrences, all_occurrences, organized, post_amt, label_names):
	words = post.split()
	scores = {}
	for word in words:
		for label in word_occurrences:
			if label not in scores:
				scores[label] = 0
			occurrences_at_label = float(word_occurrences[label].get(word, 0))
			occurrences_in_total = float(all_occurrences[word])
			if occurrences_in_total == 0.00:
				continue
			assert(occurrences_at_label is not 0)
			scores[label] += log(occurrences_at_label / occurrences_in_total)

	for label in scores:
		scores[label] += log(len(organized[label]) / float(post_amt))

	return max(scores, key=scores.get)

def write_most_freq_words(occurrences, labels):
	for label in occurrences:
		label_name = labels[label]
		with open('parsed_data/'+label_name, "w") as f:
			print 'Writing to parsed_data/'+label_name
			f.write(pprint.pformat(occurrences[label].most_common()[:200]))
			f.close()

def init_confusion_matrix(size):
	confusion_matrix = []
	for i in range(size):
		row = []
		for j in range(size):
			row.append(0)
		confusion_matrix.append(row)
	return confusion_matrix

newsgroup_posts, newsgroup_labels, label_names = load_data()
train, test, organized = construct_sets(newsgroup_posts, newsgroup_labels)
all_occurrences, word_occurrences = calculate_word_occurrences(train)

# write_most_freq_words(word_occurrences, newsgroup_data.target_names)
confusion_matrix = init_confusion_matrix(len(label_names))

correct = 0
print "Making predictions for {} items in the test set".format(len(test))
for post, expected in test:
	predicted = classify(post, word_occurrences, all_occurrences, organized, len(train), label_names)
	if predicted == expected:
		correct += 1
	confusion_matrix[expected][predicted] += 1

print pp.pprint(confusion_matrix)
print "Got {} correct predictions out of {} items, leading to {} correct predictions".format(correct, len(test), correct / float(len(test)))


