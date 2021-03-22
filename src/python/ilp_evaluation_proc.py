"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import os
import logging
import numpy as np
import json
from ilp_utils import *
from ilp_common import *
from ilp_common_classes import *

from typing import *
from typing import List, Dict

class ILPEval:
    def __init__(self, path_to_output_data: str):
        self.path_to_output_data = path_to_output_data
        self.dist_matrices_dict: Dict[str,float] = {}
        self.training_similar_set_indices_dict: Dict[str,Dict[str,float]] = {}
        self.eval_param_weights_dict: Dict[str,float] = {}
        self.eval_param_times_dict: Dict[str,float] = {}
        self.eval_param_values: List[float] = []
        self.num_vois: int
        self.training_set_size: List[int] = []

    def load_json(self,path:str) -> Dict:
        with open(os.path.join(self.path_to_output_data, path)) as f:
            var_dict = json.load(f)
        return var_dict

def eval_proc(eval_param: EvalParam, path_to_output_data: str, eval_dict: Dict[str,int]) -> tuple([np.array, np.array]):
    eval_data = ILPEval(path_to_output_data)
    eval_data.dist_matrices_dict = eval_data.load_json('voi_distances.json')
    eval_data.training_similar_set_indices_dict = eval_data.load_json('voi_similar_indices.json')
    for dir_name in list_dirs(path_to_output_data):
        weights = eval_data.load_json(os.path.join(dir_name,'voi_weight_vectors.json'))
        runtimes = eval_data.load_json(os.path.join(dir_name,'voi_runtime.json'))
        
        ilp_params_data = eval_data.load_json(os.path.join(dir_name,'ilp_params.json'))
        eval_data.num_vois = ilp_params_data['num_vois']
        eval_data.training_set_size = ilp_params_data['training_sets_size']
        eval_param_value = ilp_params_data[eval_param.value]
        
        if eval_param_value not in eval_data.eval_param_weights_dict.keys():
            eval_data.eval_param_weights_dict[str(eval_param_value)] = weights
            eval_data.eval_param_times_dict[str(eval_param_value)] = runtimes
            eval_data.eval_param_values.append(eval_param_value)
        
    mrrs = np.zeros((len(eval_data.eval_param_values), eval_data.num_vois))
    times = np.zeros((len(eval_data.eval_param_values), eval_data.num_vois))
    voi_ranks = []
    for j, node in enumerate(eval_data.dist_matrices_dict.keys()):
        training_similar_set_indices = eval_data.training_similar_set_indices_dict[str(node)]
        eval_set = eval_dict[str(node)]

        for i, eval_param_value in enumerate(eval_data.eval_param_values):

            eval_param_indx = int(i)
            node_indx = int(j)

            times[eval_param_indx][node_indx] = (eval_data.eval_param_times_dict[str(eval_param_value)])[str(node)]

            ranked_list = np.argsort(np.average(eval_data.dist_matrices_dict[str(node)], axis=1, weights=(eval_data.eval_param_weights_dict[str(eval_param_value)])[str(node)]))

            #Verify correction of results when the voi node has a ranking of 0 (since it's most similar to itself).
            logging.info('v* :{}; rank of v* :{}'.format(node, (np.where(ranked_list.astype(int) == int(node))[0][0])))
            voi_ranks.append(np.where(ranked_list.astype(int) == int(node))[0][0])

            if eval_param == EvalParam.TRAINING_SIZE:
                ranked_list_no_S = remove_S_indices([ranked_list],training_similar_set_indices[str(eval_param_value)])[0]
            else:
                ranked_list_no_S = remove_S_indices([ranked_list],training_similar_set_indices[str(eval_data.training_set_size)])[0]
            mrrs[eval_param_indx][node_indx] = mean_reciprocal_rank([ranked_list_no_S], eval_set)

    test_successful = False
    test_successful = True in (rank == 0 for rank in voi_ranks) 
    
    eval_values = {}       
    eval_values['mrr_values'] = mrrs.tolist()
    eval_values['runtime_values'] = times.tolist()
    dump_json(eval_values, os.path.join(path_to_output_data, 'eval_mrr_times.json'))
            
    return (mrrs,times,test_successful)
