from pylab import *
from random import choice
from numpy import array, dot, random, copy

unit_step = lambda x: 0 if x < 0 else 1

def classify(x):
	if x < 0:
		return -1
	else:
		return 1

def all_correct(items, w):
	correct = 0
	for x, expected in items:
		pred = classify(dot(w, x))
		if pred == expected:
			correct += 1

	return correct == len(items)

def train(items):
	train_rate = 0.5
	w = random.rand(len(items[0][0]))
	converges = False

	while not converges:
		x, expected = choice(items)
		result = dot(w, x)
		error = expected - classify(result)
		w += train_rate * error * x
		converges = all_correct(items, w)

	return w

def plot_items(items, w):
	x = []
	y = []
	color = []
	for arr, expected in items:
		x.append(arr[1])
		y.append(arr[2])
		if expected == -1:
			color.append('b')
		else:
			color.append('r')

	xlim(-1,1)
	ylim(-1,1)
	a, b = -w[1]/w[2], -w[0]/w[2]
	l = np.linspace(-1,1)
	plt.plot(l, a*l+b, c="green")
	scatter(x, y, s=100, marker='o', c=color)
	show()

training_set1 = [
	(array([1,0.3,0.3,]), 1), 
	(array([1,-0.3,0.6,]), 1), 
	(array([1,-0.3,0,]), -1), 
	(array([1,0.6,-0.9,]), -1), 
]

training_set2 = [
	(array([1,0.1,0.1, ]), 1), 
	(array([1,-0.1,-0.1, ]), -1),
	(array([1,-0.1,0.1, ]), 1),
	(array([1,0.1,-0.1, ]), -1),
]

training_set3 = [
	(array([1,0.1,0.1, ]), 1), 
	(array([1,-0.1,-0.1, ]), -1),
	(array([1,-0.1,0.1, ]), -1),
	(array([1,0.1,-0.1, ]), 1),
]

training_set4 = [
	(array([1,0.2,0.1,]), 1), 
	(array([1,0.2,0.2,]), 1), 
	(array([1,-0.2,-0.1,]), -1), 
	(array([1,-0.2,-0.3,]), -1), 
]

training_set5 = [
	(array([1,0.1,0.6,]), 1), 
	(array([1,0.1,-0.4,]), 1), 
	(array([1,0,0.3,]), -1), 
	(array([1,-0.1,0,]), -1), 
]

# The two classes are impossible to separate in a linear manner
training_set_non_converging = [
	(array([1,0.1,0.1,]), 1), 
	(array([1,-0.1,0.1,]), -1), 
	(array([1,0.1,-0.1,]), -1), 
	(array([1,-0.1,-0.1,]), 1), 
]

training_sets = [training_set1, training_set2, training_set3, training_set4, training_set5]
for training_set in training_sets:
	w = train(training_set)
	plot_items(training_set, w)