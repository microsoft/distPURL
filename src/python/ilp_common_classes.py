"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import json

from typing import *
from typing import List
from enum import Enum
from copy import copy

from ilp_common import *

class ILPError(Exception):
    '''
    ILPError Exception class.
    '''
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None
    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'ILPError has been raised'

class FileExtension(Enum):
    '''
    FileExtension Enum class: Represents different accepted file extensions for data file inputs.
    '''
    CSV     = 'csv'
    PKL     = 'pkl'
    NPY     = 'npy'
    JSON    = 'json'

class InputFormat(Enum):
    '''
    InputFormat Enum class: Represents different input format types for data file inputs.
    '''
    ONE_SET     = 'ONE SET'
    HEADER_VOIS = 'HEADER VOIS'
    COLUMN_VOIS = 'COLUMN VOIS'

class ILPCat(Enum):
    '''
    ILPCat Enum class: Represents different cateory types for ILP processing.
    '''
    INT  = 'Integer'
    CONT = 'Continuous'

class DataInputMode(Enum):
    '''
    DataInputMode Enum class: Represents different modes in which the training and eval data is generated.
    '''
    FIXED  = 'FIXED'
    RANDOM = 'RANDOM'

class RunMode(Enum):
    '''
    RunMode Enum class: Represents different modes in which code and output can run.
    '''
    DEBUG = 'DEBUG'
    RELEASE = 'RELEASE'

class EvalParam(Enum):
    '''
    EvalParam Enum class: Represents the parameter to evaluate in relation to.
    '''
    WAR = 'weight_approx_resolution'
    TRAINING_SIZE = 'training_sets_size'


class CombiningInputs:
    """
    CombiningInputs class: Contains all parameters needed by the ILP Algorithim to combine embeddings of a distance matrix.
    """
    def __init__(self):

        self.other_node_indices: List[int] = []
        self.similar_node_indices: List[int] = []
        self.dist_matrix: List[float] = []
        self.num_embeddings: int
        self.num_other_nodes: int
        self.max_dist: int
        self.up_bound: int 
        self.cat: ILPCat 
        self.solver: str 
        self.gurobi_outputflag: int
        self.gurobi_logfile: str
        self.time_limit: int
        self.num_threads: int
        self.mode: str
    
    def create_copy(self, gurobi_log_file_path: str) -> object:
        obj_copy = copy(self)
        obj_copy.gurobi_logfile = gurobi_log_file_path
        return obj_copy
    
class ILPParams:
    """
    ILPParams class contains: 
    1- All changeable parameters used for running ILP experiments.
    2- JSON representing and reading functions needed to read the class parameters.
    """

    def __init__(self):

        self.dataset: str                       = "test_set"
        self.path_to_root: str                  = "../../"
        self.path_to_metadata: str              = "../data/test_set_data/metadata.json"
        self.path_to_output: str                = "../output_data/output_test_set/"
        self.num_vois: int                      = 5
        self.training_sets_sizes: List[int]     = [10,15]
        self.minimum_ranking_thresholds: List   = [None]
        self.solvers_and_apis: List             = [["pulp","coin_cmd"]]
        self.weight_approx_resolutions: List    = [None]
        self.num_cores: int                     = 3
        self.persist_data: bool                 = True
        self.mode: RunMode                      = RunMode("DEBUG")
        self.eval_data_mode: DataInputMode      = DataInputMode("RANDOM")
        self.training_data_mode: DataInputMode  = DataInputMode("RANDOM")
        self.gurobi_outputflag: bool            = 1
        self.time_limit: int                    = 120
        self.num_threads: int                   = 1
        self.eval_size: int                     = 10
        self.eval_param: EvalParam              = EvalParam("training_sets_size")

    def __repr__(self) -> str:

        ilp_params_data = {
            "dataset"                   : self.dataset,
            "path_to_root"              : self.path_to_root,
            "path_to_metadata"          : self.path_to_metadata,
            "path_to_output"            : self.path_to_output,
            "num_vois"                  : self.num_vois,
            "training_sets_sizes"       : self.training_sets_sizes,
            "minimum_ranking_thresholds": self.minimum_ranking_thresholds,
            "solvers_and_apis"          : self.solvers_and_apis,
            "weight_approx_resolutions" : self.weight_approx_resolutions,
            "num_cores"                 : self.num_cores,
            "persist_data"              : self.persist_data,
            "mode"                      : self.mode.value,
            "eval_data_mode"            : self.eval_data_mode.value,
            "training_data_mode"        : self.training_data_mode.value,
            "gurobi_outputflag"         : self.gurobi_outputflag,
            "time_limit"                : self.time_limit,
            "num_threads"               : self.num_threads,
            "eval_size"                 : self.eval_size,
            "eval_param"                : self.eval_param.value
        }
        return json.dumps(ilp_params_data, indent=2)

    def load_data(self, path: str):
        with open(path) as f:
            data = json.load(f)
            self.dataset                    = data['dataset']
            self.path_to_root               = data['path_to_root']
            self.path_to_metadata           = data['path_to_metadata']
            self.path_to_output             = data['path_to_output']
            self.num_vois                   = data['num_vois']
            self.training_sets_sizes        = data['training_sets_sizes']
            self.minimum_ranking_thresholds = data['minimum_ranking_thresholds']
            self.solvers_and_apis           = data['solvers_and_apis']
            self.weight_approx_resolutions  = data['weight_approx_resolutions']
            self.num_cores                  = data['num_cores']
            self.persist_data               = data['persist_data']
            self.mode                       = RunMode(data['mode'].upper())
            self.eval_data_mode             = DataInputMode(data['eval_data_mode'].upper())
            self.training_data_mode         = DataInputMode(data['training_data_mode'].upper())
            self.gurobi_outputflag          = data['gurobi_outputflag']
            self.time_limit                 = data['time_limit']
            self.num_threads                = data['num_threads']
            self.eval_size                  = data['eval_size']
            self.eval_param                 = EvalParam(data['eval_param'])
