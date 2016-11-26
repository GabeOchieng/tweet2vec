'''
module: clusters.py 
use: contains functions associated clustering / unsupervised learning 
'''

import numpy as np 
from kmeans import kplusplus

def symmetrize(array): 
	'''
	Purpose: 
	Returns the symmetric version of an upper or lower triangular array

	Inputs: 
	array - upper OR lower triangular ndarray 

	Outputs: 
	symmetric version of array

	'''
	return array + array.T - np.diag(array.diagonal())

def getSimilarityArray(feature_array,similarity_method = 'exp',k_nn = 5):
	'''
	Purpose: 
	Computes the similarity array for a given feature set, similarity method, and k_nearest_neighbors value
	Part of the spectral clustering process

	Inputs: 
	feature_array - set of features 
	similarity_method - method to use for computing the similarity array: 
		--'exp' computes W[i,j] = exp(-||xi - xj||^2 / 2)
		--'norm' computes W[i,j] = ||xi - xj||^2
		--'chain' is specifically for the 'chain' generateData type
	k_nn - number of nearest neighbors to consider (k_nn=5 means only the top 5 largest similarity values are kept nonzero)

	Outputs: 
	sim_array - symmetric array of similarity strength values

	'''

	allowed_methods = ['exp','norm','chain']
	if similarity_method not in allowed_methods:
		print('ERROR: Not a valid similarity_method')
		return 
	else:
		sim_array = np.zeros((len(feature_array),len(feature_array)))
		i = 0
		j = 0
		for rowi in feature_array:
			for rowj in feature_array: 
				if i <= j: 
					difference = (rowi-rowj).T
					if similarity_method == 'exp':
						sim_array[i,j] = np.exp(-1*((difference.T).dot(difference)))
					elif similarity_method == 'norm':
						sim_array[i,j] = difference.T.dot(difference)
					elif similarity_method == 'chain':
						if np.linalg.norm(difference) <= 1.5: 
							if ((i != int(len(feature_array)/2.)-1) and (j != int(len(feature_array)/2.))):
								sim_array[i,j] = 1
							if i == j: 
								sim_array[i,j] = 1
				j += 1
			i += 1
			j = 0
		sim_array = sim_array - np.diag(sim_array.diagonal()) #remove diagonal nonzero values
		if k_nn != -1: 
			for rowi in sim_array:
				ind = np.argpartition(rowi, -1*k_nn)[(-1*k_nn):]

				for i in range(len(rowi)):
					if i not in ind: 
						rowi[i] = 0; 
		return symmetrize(sim_array)

def getDegreeArray(sim_array): #convert array W into respective Degree array, Dii = sum(i=1 to n) Wij
	'''
	Purpose: 
	Computes the Degree array 'D' in the spectral clustering process from the similarity array
	Dii = \sum_{i=1}^n Wij, ie the sum of each row of the similarity array

	Inputs: 
	sim_array - Similarity array Wij retrieved from getSimilarityArray()

	Outputs: 
	D - degree array (described in Purpose

	'''
	D = np.zeros((sim_array.shape[0],sim_array.shape[0]))
	for i in range(0,sim_array.shape[0]):
		D[i,i] = np.sum(sim_array[i,:])
	return D

def getLaplacian(W,D): 
	'''
	Purpose: 
	Returns the Laplacian of the similarity array W and the degree array D 
	For use with spectral clustering

	Inputs: 
	W - similarity array from getSimilarityArray()
	D - degree array from getDegreeArray

	Outputs: 
	L = D-W, the laplacian 

	'''
	return D-W

def getLaplacianBasis(features,similarity_method='exp',k_nn=5, get_W=True): 
	'''
	Purpose: 
	Returns orthogonal basis for Laplacian embedding of features. Essentially the full spectral clustering algorithm before the actual clustering

	Inputs: 
	features - n examples by k features ndarray (n>k preferred)
	similarity_method - method to use for computing the similarity array: 
		--'exp' computes W[i,j] = exp(-||xi - xj||^2 / 2)
		--'norm' computes W[i,j] = ||xi - xj||^2
		--'chain' is specifically for the 'chain' generateData type
	k_nn - number of nearest neighbors to consider in similarity array
	num_clusters - number of clusters for kmeans++ to sort the data into
	get_W - if get_W is False, then features is actually the similarity matrix W, and thus a new W will not be computed according to similarity_method

	Outputs: 
	U - orthogonal basis returned by the svd of the laplacian with columns corresponding to the most significant singular values at the lowest indices

	'''
	if get_W: 
		W = getSimilarityArray(features,similarity_method,k_nn)
	else: 
		W = features
	D = getDegreeArray(W)
	L = getLaplacian(W,D)
	U,s,V = np.linalg.svd(L,full_matrices=0)
	return U

def spectralClustering(features,similarity_method='exp',k_nn=5,basis_dim=2,num_clusters=2,get_W=True): 
	'''
	Purpose: 
	Performs spectral clustering into 'num_clusters' clusters on data defined in the ndarray 'features'

	Inputs: 
	features - n examples by k features ndarray (n>k preferred)
	similarity_method - method to use for computing the similarity array: 
		--'exp' computes W[i,j] = exp(-||xi - xj||^2 / 2)
		--'norm' computes W[i,j] = ||xi - xj||^2
		--'chain' is specifically for the 'chain' generateData type
	k_nn - number of nearest neighbors to consider in similarity array
	basis_dim - number of svd basis vectors to consider for input to kmeans++ algorithm
	num_clusters - number of clusters for kmeans++ to sort the data into
	get_W - if get_W is False, then features is actually the similarity matrix W, and thus a new W will not be computed according to similarity_method

	Outputs: 
	labels - 1 by n array of assigned cluster labels for each feature example
	centers - cluster centers array (basis_dim by num_clusters) representing each of the k cluster centers 

	'''

	#W = getSimilarityArray(features,similarity_method,k_nn)
	#D = getDegreeArray(W)
	#L = getLaplacian(W,D)
	#U,s,V = np.linalg.svd(L,full_matrices=0)
	U = getLaplacianBasis(features=features,similarity_method=similarity_method,k_nn=k_nn,get_W=get_W)
	U = U[:,-1*basis_dim:]
	labels, centers = kplusplus(U.T,num_clusters)
	return labels, centers, U 
