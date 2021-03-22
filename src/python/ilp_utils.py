"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import numpy as np
import seaborn as sns

def mean_reciprocal_rank(rank_lists, inds):
    """
    Calculates mean reciprocal rank given a rank list and set of indices of interest.
    
    Input
    rank_list - array-like (length=n-1)
        A ranked array-like of objects (assumed to be integers).
    inds - array-like (length<=n-1)
        A array-like of objects (assumed to be integers).
        
    Return
    mrr - float
        Mean reciprocal rank of the objects in inds.
    """
    mrrs = np.zeros(len(rank_lists))
    
    for i, r in enumerate(rank_lists):
        mrrs[i] = np.mean(1 / (np.array([np.where(r == s)[0][0] for s in inds])))    
        
    return mrrs

def edit_dist_matrices(dist_matrices, s_stars, ranks, threshold=500):

    editted_dist_matrices = dist_matrices.copy()
    arg_sorts = np.argsort(dist_matrices, axis=0).T
    ranks_below_threshold_by_representation = get_vertices_below_threshold(ranks, threshold)
    
    for i, s in enumerate(s_stars):
        for j, s_rank in enumerate(ranks[i]):
            if s_rank > threshold:
                if len(ranks_below_threshold_by_representation[j]) == 0:
                    sampling_array = np.arange(threshold)
                else:
                    sampling_array = ranks_below_threshold_by_representation[j]

                temp_rank = int(np.random.choice(sampling_array, 1)[0])
                editted_dist_matrices[s, j] = dist_matrices[arg_sorts[j][temp_rank], j] + (1e-10 / s)
                
    return editted_dist_matrices

def remove_S_indices(rank_lists, S_indices):
    """
    A function to remove elements from a rank-list.
    
    Input
    rank_lists - list
        A list of arrays.
    S_indices - array-like
        The set of objects to be removed.
        
    Return
    new_rank_lists - list
        A list of arrays with S_indices removed.
    """
    
    new_rank_lists = []
    for i, r in enumerate(rank_lists):
        idx = np.array([np.where(r == s)[0][0] for s in S_indices])
        new_rank_lists.append(np.delete(r, idx))
        
    return new_rank_lists
 
def evaluate_best_vertices(dist_matrix, vertices, s_star):
    """
    A function to evaluate a set of individual metrics.
    
    Input
    dist_matrix - array (shape=(n,J))
        An array containing J distances from an object of interest to n-1 other objects.
    vertices - array-like
        The set of individual metrics to evaluate.
    s_star - array-like
        The set of indices for which the individual metrics are evaluated.
        
    Return
    ranks - np.array
        The rankings of the elements of s_star.
    """
    
    ranks = np.zeros((len(s_star), len(vertices)))
    for i, s in enumerate(s_star):
        for j, v in enumerate(vertices):
            temp_ranks = np.argsort(dist_matrix[:, j])
            ranks[i, j] = np.array([np.where(temp_ranks == s)[0][0]])

    return ranks
