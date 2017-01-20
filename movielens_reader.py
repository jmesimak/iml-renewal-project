import pprint

# ratings: a numpy array where each rows is (userID, movieID, rating, timestamp)
import numpy as np
from datasets.movielens import loadmovielens as reader

pp = pprint.PrettyPrinter(indent=4)
TOY_STORY_ID = 1
GOLDEN_EYE_ID = 2

CHOSEN_MOVIES = [1, 2, 23, 56, 62, 69, 71, 82, 89, 135, 156, 271, 405, 475, 820, 894, 362, 175, 272, 423]

def read_movielens():
	return reader.read_movie_lens_data()

ratings, movie_dictionary, user_ids, item_ids, movie_names = read_movielens()
print "read the data"


def calc_jaccard(movie1, movie2):
	ratings_movie_ids = np.array([rating[1] for rating in ratings])
	m1ids = np.where(ratings_movie_ids == movie1)[0]
	m2ids = np.where(ratings_movie_ids == movie2)[0]
	m1users = np.array([rating[0] for rating in ratings[m1ids]])
	m2users = np.array([rating[0] for rating in ratings[m2ids]])
	same = len(np.intersect1d(m1users, m2users))
	return same / float(len(m1users) + len(m2users) - same)

def calc_matrix(movies):
	coeffs = []
	ctr = 0
	for idx1, id1 in enumerate(movies):
		coeffs.append([])
		for idx2, id2 in enumerate(movies):
			jacc = calc_jaccard(id1, id2)
			coeffs[idx1].append(jacc)

		ctr += 1
		print "{} out of {}".format(ctr, len(movies))

	return coeffs

jacc_coeffs = calc_matrix(CHOSEN_MOVIES)

def mov_clusters(movie_ids):
	clusters = []
	for movie_id in movie_ids:
		clusters.append([(movie_id, movie_dictionary.get(movie_id))])
	return clusters

def print_jacc(mtx):
	for idx1, m1 in enumerate(mtx):
		for idx2, m2 in enumerate(mtx):
			print "{} has a jaccard coefficient of {} with {}".format(movie_dictionary[CHOSEN_MOVIES[idx1]], mtx[idx1][idx2], movie_dictionary[CHOSEN_MOVIES[idx2]])

def get_movie_distance(m1, m2):
	idx1 = CHOSEN_MOVIES.index(m1[0])
	idx2 = CHOSEN_MOVIES.index(m2[0])
	return jacc_coeffs[idx1][idx2]

def single_linkage_distance(c1, c2):
	best_distance = get_movie_distance(c1[0], c2[0])
	for movie1 in c1:
		for movie2 in c2:
			d = get_movie_distance(movie1, movie2)
			if d > best_distance:
				best_distance = d
	return best_distance

def calc_dists(clusters):
	distances = []
	for idx1, c1 in enumerate(clusters):
		distances.append([])
		for idx2, c2 in enumerate(clusters):
			distances[idx1].append(single_linkage_distance(c1, c2))
	return distances

def get_best_match_indices(cluster_distances):
	best_match = (0, 1)
	best_distance = cluster_distances[0][1]
	for idx1, c1 in enumerate(cluster_distances):
		for idx2, c2 in enumerate(cluster_distances):
			d = cluster_distances[idx1][idx2]
			if d > best_distance and idx1 is not idx2:
				best_match = (idx1, idx2)
				best_distance = d
	return best_match, best_distance

def merge(c1, c2, clusters):
	print "Merging the following movies"
	new_cluster = []
	for m in c1:
		print "\t{}".format(m[1])
		new_cluster.append(m)
	for m in c2:
		print "\t{}".format(m[1])
		new_cluster.append(m)
	clusters.remove(c1)
	clusters.remove(c2)
	clusters.append(new_cluster)
	return clusters

def clusterize(clusters):
	while len(clusters) > 1:
		distances = calc_dists(clusters)
		best_match, best_distance = get_best_match_indices(distances)
		merge(clusters[best_match[0]], clusters[best_match[1]], clusters)



clusters = mov_clusters(CHOSEN_MOVIES)
clusterize(clusters)

# movies = movie_dictionary.values()

# print_jacc(jacc_mtx)

