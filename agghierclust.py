import math
import pprint

pp = pprint.PrettyPrinter(indent=4)

class Point:
	def __init__(self, location, label):
		self.location = location
		self.label = label

def create_points(locations):
	points = []
	ch = 'a'
	for location in locations:
		points.append(Point(location, ch))
		ch = chr(ord(ch)+1)
	return points

def pairwise_negative_euclidean(p1, p2):
	x1 = p1[0]
	x2 = p2[0]
	y1 = p1[1]
	y2 = p2[1]
	first = math.pow(x1-x2, 2)
	second = math.pow(y1-y2, 2)
	return -math.sqrt(first + second)

def dummy_distance(p1, p2):
	return abs(p2.location - p1.location)

def single_linkage_distance(cluster1, cluster2, distance_fn):
	minDist = distance_fn(cluster2[0],cluster1[0])
	for p1 in cluster1:
		for p2 in cluster2:
			dist = distance_fn(p1, p2)
			if dist < minDist:
				minDist = dist
	return minDist

def complete_linkage_distance(cluster1, cluster2, distance_fn):
	maxDist = distance_fn(cluster2[0], cluster1[0])
	for p1 in cluster1:
		for p2 in cluster2:
			dist = distance_fn(p1, p2)
			if dist > maxDist:
				maxDist = dist
	return maxDist

def calculate_distance_matrix(clusters, linkage_fn, distance_fn):
	distances = []
	for idx1, c1 in enumerate(clusters):
		distances.append([])
		for idx2, c2 in enumerate(clusters):
			if c1 == c2:
				distances[idx1].append(0)
			else:
				d = abs(linkage_fn(c1, c2, distance_fn))
				distances[idx1].append(d)

	return distances

def get_min_indices(distance_matrix):
	smallest = distance_matrix[0][1]
	minRow, minCol = 0, 1
	for idxRow, row in enumerate(distance_matrix):
		for idxCol, col in enumerate(distance_matrix):
			cur = distance_matrix[idxRow][idxCol]
			if cur is not 0 and cur < smallest:
				smallest = cur
				minRow, minCol = idxRow, idxCol
	print "Merging two clusters at distance {}".format(smallest)
	return minRow, minCol, smallest

def clusterize_points(points):
	return [[point] for point in points]

def print_cluster(cluster):
	print "\tCluster has the following points"
	for p in cluster:
		print "\t\t{}: {}".format(p.label, p.location)	

def merge_clusters(clusters, cluster1, cluster2):
	cluster = []
	for c1 in cluster1:
		cluster.append(c1)
	for c2 in cluster2:
		cluster.append(c2)
	clusters.remove(cluster1)
	clusters.remove(cluster2)
	clusters.append(cluster)
	return clusters

def cluster(clusters):
	linkage = "single"
	distance = "negative euclidean"
	print "Running agglomerative hierarchical clustering with {} linkage while using {} distance measure".format(linkage, distance)
	linkages = {
		"single": single_linkage_distance,
		"complete": complete_linkage_distance
	}

	distance_measures = {
		"dummy": dummy_distance,
		"negative euclidean": pairwise_negative_euclidean
	}

	link_fn = linkages[linkage]
	distance_fn = distance_measures[distance]

	while len(clusters) > 1:
		distances = calculate_distance_matrix(clusters, link_fn, distance_fn)
		minRow, minIdx, minValue = get_min_indices(distances)
		c1 = clusters[minRow]
		c2 = clusters[minIdx]
		print "Merging"
		print_2d_cluster(c1)
		print_2d_cluster(c2)
		# print_cluster(c1)
		# print_cluster(c2)
		clusters = merge_clusters(clusters, c1, c2)


	# distances[minRow]
	# pp.pprint(distances)

def print_2d_cluster(c):
	print "Cluster contains the following points:"
	for p in c:
		print "\t({}, {})".format(p[0],p[1])

def construct_two_dimensional_pts():
	return [
		[(1,1)],
		[(3,8)],
		[(8,1)],
		[(10,8)],
		[(12,11)]
	]

points = create_points([1,2,4,7,8,12,13,14,16,18,20])
clusters = clusterize_points(points)
clusters_2d = construct_two_dimensional_pts()
cluster(clusters_2d	)


# strs = ["{}: {}".format(point.label, point.location) for point in points]
# print strs

