"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

# import open-source apis (pulp, mip, gurobi)
import gurobipy as gp
from gurobipy import Model
import pulp
import mip

# import some utility functions
from ilp_utils import *
from ilp_common_classes import *

#import other python libraries
import numpy as np
from tqdm import tqdm
from colorama import Fore
import time
from typing import *
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def using_api_pulp(inputs: CombiningInputs) -> np.array:
    '''
    Performs ILP processing using PuLP API to return an np.array of a weights vector for representations to be combined.
    '''
    # Define a model.
    model = pulp.LpProblem(sense=pulp.LpMinimize)
    
    # Define indicator variables for every vertex that is not in inputs.similar_node_indices.
    dict_lpvariables_node_indices = pulp.LpVariable.dicts("indicators for elements not in similar_nodes_indices", 
                                (node for node in inputs.other_node_indices),
                                cat='Integer',
                                upBound=1,
                                lowBound=0
                                )

    logging.info("indicators defined")
    # Define non-negative weight variables for each of the representations.
    # inputs.up_bound = 100 if weight_approx_resolution = .01 and so on....
    dict_lpvariables_embed_weights = pulp.LpVariable.dicts("weights for representations",
                                    (j for j in range(inputs.num_embeddings)),
                                    cat=inputs.cat.name,
                                    upBound=inputs.up_bound,
                                    lowBound=0
                                    )

    logging.info("weight variable defined")
    # Set the objective function.
    model += (
        pulp.lpSum(
            [dict_lpvariables_node_indices[(node)] for node in inputs.other_node_indices]
        )
    )

    # Add constraint that the weights must sum to the upper bound defined by weight_approx_resolution.
    model += (
        pulp.lpSum(
            [dict_lpvariables_embed_weights[(j)] for j in range(inputs.num_embeddings)]
        ) == inputs.up_bound
    )
    logging.info("objective function defined")
    # Add constraint that elements of inputs.similar_node_indices should be closer than elements not in inputs.similar_node_indices (or, other_node_indices)
    for s in tqdm(inputs.similar_node_indices, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
        for node in tqdm(inputs.other_node_indices):
            model += (
                pulp.lpSum(
                    [dict_lpvariables_embed_weights[(j)] * inputs.dist_matrix[s, j] for j in range(inputs.num_embeddings)]
                )
                <=
                pulp.lpSum(
                    [dict_lpvariables_embed_weights[(j)] * inputs.dist_matrix[node, j] for j in range(inputs.num_embeddings)]
                ) 
                + 
                pulp.lpSum(dict_lpvariables_node_indices[(node)] * inputs.max_dist * inputs.up_bound)
            )
    logging.info("constraints defined")
    
    if inputs.solver == 'pulp':
        logging.info("Call pulp solver")
        model.solve()
        logging.info("pulp solver is done")
    elif inputs.solver == 'coin_cmd':
        model.solve(solver=pulp.COIN_CMD(msg= True,timeLimit= inputs.time_limit,threads= inputs.num_threads))
        logging.info("coin_cmd solver is done, timeLimit = %s, threads = %s" % (inputs.time_limit, inputs.num_threads))
    elif inputs.solver == "glpk":
        logging.info("Call glpk solver")
        model.solve(solver=pulp.GLPK_CMD())
        logging.info("glpk solver is done")
    elif inputs.solver == 'gurobi_cmd':
        model.solve(solver=pulp.GUROBI_CMD(msg = True, timeLimit= inputs.time_limit, threads= inputs.num_threads))
        logging.info("gurobi_cmd solver is done")
        logging.info("coin_cmd solver is done,timeLimit = %s, threads = %s" % (inputs.time_limit, inputs.num_threads))

    alpha_hat = np.array([w.varValue for w in dict_lpvariables_embed_weights.values()])
    return alpha_hat
   

def using_api_gurobi(inputs: CombiningInputs) -> np.array:
    '''
    Performs ILP processing using Gurobi API to return an np.array of a weights vector for representations to be combined.
    '''
    start_time = time.time()
    
    #create an ILP model
    model = Model()
    
    #set Gurobi parameters
    model.setParam('OutputFlag', inputs.gurobi_outputflag)
    model.setParam(gp.GRB.Param.Threads, inputs.num_threads)
    model.setParam('TimeLimit', inputs.time_limit)
    if inputs.mode == RunMode.DEBUG:
        model.setParam('LogFile',inputs.gurobi_logfile, 'a')
    model.setParam(gp.GRB.Param.LogToConsole, False)

    logging.info("Gurobi's parameters are set")

    # Define indicator variables for every vertex that is not in inputs.similar_node_indices.
    indicators_tupledict = model.addVars(inputs.num_other_nodes, vtype=gp.GRB.BINARY, name='indicators_tupledict')
    model.setObjective(gp.quicksum(indicators_tupledict), gp.GRB.MINIMIZE)
    
    #set weight vector catergory
    if inputs.cat == ILPCat.CONT:
        v_type = gp.GRB.CONTINUOUS

    elif inputs.cat == ILPCat.INT:
        v_type = gp.GRB.INTEGER
    
    #Define non-negative weight variables for each of the representations.
    weights_tupledict = model.addVars(inputs.num_embeddings, lb=0, ub=inputs.up_bound,
                        vtype=v_type, name='weights_tupledict')
   
    logging.info("Done with model variable definitions and start adding constraints...")

    #Add constraint that the weights must sum to the upper bound defined by weight_approx_resolution.
    model.addConstr(weights_tupledict.sum() == inputs.up_bound)

    #set up inequalities: that elements of inputs.similar_node_indices should be closer than elements not in inputs.similar_node_indices (or, other_node_indices)
    right_side_constr_inequality = []
    
    for n, node in enumerate(tqdm(inputs.other_node_indices, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.YELLOW, Fore.RESET))):
        other_node_distance = (n,dict([((i), inputs.dist_matrix[node, i])
                                for i in range(inputs.num_embeddings)]))
        right_side_constr_inequality.append((weights_tupledict.prod(other_node_distance[1]))+
                                            (indicators_tupledict[other_node_distance[0]]*inputs.max_dist*inputs.up_bound))

    #Add inequalities as constraints 
    for s in tqdm(inputs.similar_node_indices, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.RED, Fore.RESET)):
        similar_node_distance = dict([((i), inputs.dist_matrix[s, i])
                                for i in range(inputs.num_embeddings)])
        left_side_constr_inequality = weights_tupledict.prod(similar_node_distance)
        for ct in tqdm(right_side_constr_inequality):
            model.addConstr(left_side_constr_inequality <= ct)

    logging.info("Constraints added... ")

    #optimize/solve the model
    model.optimize()
    alpha_hat = np.array([i.X for i in list(weights_tupledict.values())])

    logging.info("Gurobi solver is done with %s thread(s)" % (inputs.num_threads))
    elapsed_time = time.time() - start_time
    if inputs.mode == RunMode.DEBUG:
        with open(inputs.gurobi_logfile, 'a') as log_file:
            log_file.write("\n\nRuntime for current node: "+str(elapsed_time/60)+" minutes.\n\n")
        
    return alpha_hat


def using_api_py_mip(inputs: CombiningInputs) -> np.array:
    '''
    Performs ILP processing using Py-MIP API to return an np.array of a weights vector for representations to be combined.
    '''
    #create an ILP model
    model = mip.Model(sense=mip.MINIMIZE)

    #Define indicator variables for every vertex that is not in inputs.similar_node_indices.
    dict_lpvariables_node_indices = [model.add_var(
        name='dict_lpvariables_node_indices', var_type=mip.BINARY) for node in range(inputs.num_other_nodes)]

    #Define non-negative weight variables for each of the representations.
    dict_lpvariables_embed_weights = [model.add_var(
        name='dict_lpvariables_embed_weights', lb=0.0, ub=inputs.up_bound, var_type = inputs.cat.name) for j in range(inputs.num_embeddings)]

    #Add constraint that the weights must sum to the upper bound defined by weight_approx_resolution.
    model += mip.xsum(w for w in dict_lpvariables_embed_weights) == inputs.up_bound

    # Add constraint that elements of inputs.similar_node_indices should be closer than elements not in inputs.similar_node_indices (or, other_node_indices)
    for s in inputs.similar_node_indices:
        for i, node in enumerate(inputs.other_node_indices):
            model += mip.xsum(dict_lpvariables_embed_weights[j] * inputs.dist_matrix[s, j] for j in range(inputs.num_embeddings)) <= mip.xsum(
                dict_lpvariables_embed_weights[j] * inputs.dist_matrix[node, j] for j in range(inputs.num_embeddings)) + dict_lpvariables_node_indices[i]*inputs.max_dist*inputs.up_bound

    #solve/optimize model
    model.objective = mip.xsum(
        indicators_tupledict for indicators_tupledict in dict_lpvariables_node_indices)
    model.optimize(max_seconds=inputs.time_limit)

    alpha_hat = np.array([w.x for w in dict_lpvariables_embed_weights])
    return alpha_hat

def using_api(api: str, inputs: CombiningInputs) -> np.array:
    # Each API is relatively similar. First, you define a set of variables. 
    # Then, using these variables, you define an objective function and a set of constraints that you assign to a model object. 

    if api == 'pulp':
        try:
            alpha_hat = using_api_pulp(inputs)
            logging.info("done with pulp") 
        except Exception as e:
            logging.error(e)
            raise e
               
    elif api=='gurobi':
        try:
            logging.info("starting Gurobi")
            alpha_hat = using_api_gurobi(inputs)           
            logging.info("done with gurobi")
        except Exception as e:
            logging.error("Exeception occured: %s" % e)
            raise e
        
    elif api=='py-mip':
        try:
            alpha_hat = using_api_py_mip(inputs)
            logging.info("done with mip")
        except Exception as e:
            logging.error(e)
            raise e
    
    else:
        raise ILPError("api %s not implemented"%(api))
    return alpha_hat

def get_similar_and_other_nodes(dist_matrix: np.array, key_voi_index: int, similar_node_indices: np.array, ilp_params: ILPParams) -> CombiningInputs:
    """
    Input:
    dist_matrix - np.array (shape=(num_nodes, num_embeddings))
        Array containing the distances between the vertex of interest and the other num_nodes - 1
        vertices.
    key_voi_index - int
        Index of vertex of interest.
    similar_node_indices - array-like
        Indices of the vertices that should be at the top of the
        nomination list for the vertex of interest.
    Returns:
    CombiningInputs Object with its parameters partially defined.
    """
    # Grab the shape of the distance matrix
    num_nodes, num_embeddings = dist_matrix.shape

    # Grab the maximum value of dist_matrix (to be used as an upper bound later)
    max_dist = np.max(abs(dist_matrix))

    # Grab the number of elements known to be similar to key_voi_index
    num_similar_nodes = len(similar_node_indices)
    logging.info("num of similar nodes: %s, num of embeddings: %s, num of nodes %s" %
          (num_similar_nodes, num_embeddings, num_nodes))

    # Define an array of integers corresponding to elements not in similar_node_indices
    other_node_indices = np.array([int(i) for i in np.concatenate((range(
        0, key_voi_index), range(key_voi_index+1, num_nodes))) if i not in similar_node_indices])

    # Grab the number of elements not known to be similar to key_voi_index
    num_other_nodes = len(other_node_indices)

    #Create a CombiningInputs Object:
    inputs = CombiningInputs()
    #define the some of the parameters of the object:
    inputs.other_node_indices = other_node_indices
    inputs.similar_node_indices = similar_node_indices
    inputs.dist_matrix = dist_matrix
    inputs.num_embeddings = num_embeddings
    inputs.num_other_nodes = num_other_nodes
    inputs.max_dist = max_dist
    inputs.gurobi_outputflag = ilp_params.gurobi_outputflag
    inputs.time_limit = ilp_params.time_limit
    inputs.num_threads = ilp_params.num_threads
    inputs.mode = ilp_params.mode

    return inputs

def combine_representations(voi_experiment: object) -> np.array:
    """
    A function to find the dict_lpvariables_embed_weights of optimal linear combination of representations. Returns normalized weight vectors.
    """
    # Pre-process the data so that there are no elements of similar_node_indices that are above threshold
    if voi_experiment.mrr is not None:
        # get ranks of 4 similar products (4 x 1 array)
        ranks = evaluate_best_vertices(voi_experiment.combining_inputs.dist_matrix, vertices=np.arange(voi_experiment.combining_inputs.num_embeddings), s_star=voi_experiment.combining_inputs.similar_node_indices)

        # get an edited distance matrix
        dist_matrix = edit_dist_matrices(voi_experiment.combining_inputs.dist_matrix, voi_experiment.combining_inputs.similar_node_indices, ranks, voi_experiment.mrr)
    
    # We can either use continuous a weight vector for combining the representations or a discrete approximation
    
    if voi_experiment.war is not None:
        # Here, we are using a discrete approximation
        
        # weight_approx_resolution is in the interval (0, 1]. If weight_approx_resolution is close to 0 that means 
        # we want our approximation to be of a high resolution. To achieve this, we define 1 / weight_approx_resolution
        # to be the maximum value that a particular weight can take on. 
        # i.e. with weight_approx_resolution = 0.1, the dict_lpvariables_embed_weights can take on values 0.0, 0.1, 0.2, 0.3, .., 1.
        #we will get (1/weight_approx_resolution) possible weights, which is the up_bound.
        # We normalize later.
        up_bound = int(np.math.ceil(1 / voi_experiment.war))
        cat = ILPCat.INT
    else:
        # Here, we let the dict_lpvariables_embed_weights be continuous
        up_bound=1
        cat= ILPCat.CONT
    logging.info("cat: %s and up_bound: %s" % (cat.name, up_bound))  

    #define the api parameters of the CombiningInputs object:
    voi_experiment.combining_inputs.up_bound = up_bound
    voi_experiment.combining_inputs.cat = cat
    voi_experiment.combining_inputs.solver = voi_experiment.solver
    logging.info("Done with all definitions for combine_representations()")

    try:
        alpha_hat = using_api(voi_experiment.api, voi_experiment.combining_inputs)
        
    except Exception as e:
        logging.error(e)
        return None
    
    #if returned weights vectors are empty
    if alpha_hat[0] is None or len(alpha_hat) == 0:
        return None
  
    else:
        # Normalize
        logging.info("Normalizing...")
        return alpha_hat / np.sum(alpha_hat)
 
