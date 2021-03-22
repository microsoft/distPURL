"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import numpy as np
from tqdm import tqdm
import logging
from ilp_common import *
from ilp_metadata_proc import *
from ilp_common_classes import *

from typing import *
from typing import List, Dict

class ILPPreProcData:
    '''
    ILPPreProcData class: Contains all data necessary for the ILP pipeline in an organized dict and list format.
    '''
    def __init__(self):
        logging.info("start pre-processing Data")
        self.v_stars: List[int] = []
        self.testing_similar_sets_indices: Dict[str,int] = {}
        self.training_dict: Dict[str, int] = {}
        self.eval_dict: Dict[str, int] = {}
        self.dist_matrices: Dict[str, float] = {}
        
    def get_vstars(self, all_vois: List[int], ilp_params: ILPParams, indices: List[int]):
        '''
        Load v_stars data
        '''
        #grab all v_stars at random of size ilp_params.num_vois
        if len(indices) == 0:
            v_stars = np.random.choice(
                all_vois, ilp_params.num_vois, replace=False)
        else:
            v_stars = np.asarray(indices).astype(int)

        self.v_stars = v_stars.tolist()

    def get_dist_matrices(self, all_vois: List[int], all_data: ilp_metadata_json_parser):
        '''
        Load distance_matrices data
        '''
        if not self.v_stars:
            raise ILPError('v_stars needs to be initialized first')
        
        dist_matrices = all_data.get_distance_matrices()
        dist_matrices_dict = {}
        print("Dist Matrices shape: ", dist_matrices.shape)
        # create an edited copy of all the vertices of interest without v*
        for i in tqdm(self.v_stars):
            dist_matrices_dict[str(int(i))] = dist_matrices[np.where(
                all_vois == i)[0][0]].tolist()

        self.dist_matrices = dist_matrices_dict

    def get_testing_similar_sets_indices(self, all_vois: List[int], all_data: ilp_metadata_json_parser):
        '''
        Load testing_similar_sets_indices data
        '''
        if not self.v_stars:
            raise ILPError('v_stars needs to be initialized first')
        
        #testing_data: dict
        testing_data = all_data.get_testing_set()
        if type(testing_data) == dict: #dict
            for i in tqdm(self.v_stars):
                self.testing_similar_sets_indices[str(int(i))] = testing_data[str(int(i))].tolist()  
                self.testing_similar_sets_indices[str(int(i))] = [int(j) for j in self.testing_similar_sets_indices[str(int(i))] if not np.isnan(j)]
        elif type(testing_data) == np.array or type(testing_data) == np.ndarray or type(testing_data) == list:  # ONE SET
            for i in tqdm(self.v_stars):
                self.testing_similar_sets_indices[str(int(i))] = np.delete(
                    testing_data, np.where(all_vois == i)[0][0]).tolist()
        else:
            self.testing_similar_sets_indices = None
    def get_eval_dict(self, all_vois: List[int], ilp_params: ILPParams, all_data: ilp_metadata_json_parser):
        '''
        Load eval_dict data
        '''
        if not self.v_stars:
            raise ILPError('v_stars needs to be initialized first')

        #eval_data is a dict or None
        if ilp_params.eval_data_mode == DataInputMode.FIXED:
            eval_data = all_data.get_eval_set()
            if eval_data is  None:
                raise ILPError("Insufficient data, please provide an eval_data_file or use the RANDOM eval_data_mode. See manual for more details.")
            else:
                for i in tqdm(self.v_stars):
                    self.eval_dict[str(int(i))] = eval_data[str(int(i))].tolist()
                    self.eval_dict[str(int(i))] = [int(j) for j in self.eval_dict[str(int(i))] if not np.isnan(j)]

        elif ilp_params.eval_data_mode == DataInputMode.RANDOM:
            if not self.testing_similar_sets_indices:
                raise ILPError('testing_similar_sets_indices needs to be initialized first')
            for i in tqdm(self.v_stars): #eval_dict is chosen after training_dict
                if ilp_params.training_data_mode == DataInputMode.FIXED:
                    try:
                        self.eval_dict[str(int(i))] = np.random.choice([int(s) for s in self.testing_similar_sets_indices[str(int(i))] if (s not in self.training_dict[str(int(i))] and not np.isnan(s))], ilp_params.eval_size, replace=False).tolist()
                    except ValueError as e:
                        raise Exception("Choose a smaller eval_size, it's too high, allowing for an overlap between training and evaluation data. See manual for more details.") from e
                else:#eval_dict is chosen before training_dict
                    self.eval_dict[str(int(i))] = np.random.choice(self.testing_similar_sets_indices[str(int(i))], ilp_params.eval_size, replace=False).tolist()
        else:
            raise ILPError("eval_data_mode is not supported, please use FIXED or RANDOM, see manual for more details.")

    def get_training_dict(self, all_vois: List[int], ilp_params: ILPParams, all_data: ilp_metadata_json_parser):
        '''
        Load training_dict data
        '''
        if not self.v_stars:
            raise ILPError('v_stars needs to be initialized first')

        #training_data is a dict or None
        if ilp_params.training_data_mode == DataInputMode.FIXED:
            training_data = all_data.get_training_set()
            if training_data is  None:
                raise ILPError("Insufficient data, please provide an training_data_file or use the RANDOM training_data_mode. See manual for more details.")
            else:
                for i in tqdm(self.v_stars):
                    self.training_dict[str(int(i))] = training_data[str(int(i))].tolist()
                    self.training_dict[str(int(i))] = [int(j) for j in self.training_dict[str(int(i))] if not np.isnan(j)]

        elif ilp_params.training_data_mode == DataInputMode.RANDOM:
            #training_sets chosen by the driver from testing_similar_sets_indices according to all training_sets_sizes
            self.training_dict = None
        else:
            raise ILPError("training_data_mode is not supported, please use FIXED or RANDOM, see manual for more details.")

def validate_no_overlap(training_dict: Dict, eval_dict: Dict):
    for i in training_dict.keys():
        if not set(training_dict[i]).isdisjoint(eval_dict[i]): 
            raise ILPError('evaluation data and training data cannot overlap!')

def ilp_pre_proc(ilp_params: ILPParams, all_data: ilp_metadata_json_parser, indices:List[int] = []) -> ILPPreProcData:
    all_vois = all_data.get_vois()

    data = ILPPreProcData()
    data.get_vstars(all_vois, ilp_params, indices)
    data.get_dist_matrices(all_vois, all_data)
    data.get_testing_similar_sets_indices(all_vois, all_data)
    if ilp_params.training_data_mode == DataInputMode.FIXED:
        data.get_training_dict(all_vois, ilp_params, all_data)
        data.get_eval_dict(all_vois, ilp_params, all_data)
        validate_no_overlap(data.training_dict,data.eval_dict)
    else:
        data.get_eval_dict(all_vois, ilp_params, all_data)
        data.get_training_dict(all_vois, ilp_params, all_data)
        
    return data
