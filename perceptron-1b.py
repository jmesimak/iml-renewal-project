from matplotlib import *
from random import choice
from numpy import array, dot, random, copy

from datasets.mnist import mnist_load_show as mnist

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

def load_images():
	d_size = 5000
	X, y = mnist.read_mnist_training_data(d_size)
	training_imgs = X[:d_size/2]
	training_labels = y[:d_size/2]
	test_imgs = X[d_size/2:]
	test_labels = y[d_size/2:]
	return training_imgs, training_labels, test_imgs, test_labels

def convert(images, labels):
	converted = []
	for idx, label in enumerate(labels):
		if label == 1:
			converted.append((images[idx], label))
		else:
			converted.append((images[idx], -1))

	return converted

def test_perceptron(weights, test_set):
	corr = 0
	for mtx, expected in test_set:
		res = classify(dot(weights, mtx))
		if res == expected:
			corr += 1

	print 'Correct of test set'
	print str((float(corr) / float(len(test_set))) * 100) + '%'

def train(items):
	train_rate = 0.1
	w = random.rand(len(items[0][0]))
	converges = False
	iters = 0
	while not converges:
		iters += 1
		x, expected = choice(items)
		result = dot(w, x)
		error = expected - classify(result)
		w += train_rate * error * x
		converges = all_correct(items, w)

	print iters
	mnist.visualize(w)
	return w

def filter_images(training_imgs, training_labels, test_imgs, test_labels):
	training_set = convert(training_imgs, training_labels)
	test_set = convert(test_imgs, test_labels)
	w = train(training_set)
	test_perceptron(w, test_set)	


training_imgs, training_labels, test_imgs, test_labels = load_images()
filter_images(training_imgs, training_labels, test_imgs, test_labels)
