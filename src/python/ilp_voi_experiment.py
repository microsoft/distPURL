"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import time
from typing import *

from ilp_common import *
from ilp_processing import *

class ILPVoiExperiment:
    def __init__(self, v_star, training_sets_size, api, solver, mrr, war, combining_inputs):
        self.hash: str = str(v_star)+'_'+str(training_sets_size)+'_'+str(api)+'_'+str(solver)+'_'+str(mrr)+'_'+str(war)
        self.v_star: int = v_star
        self.training_sets_size: int = training_sets_size
        self.api: str = api
        self.solver: str = solver
        self.mrr: float = mrr
        self.war: float = war
        self.combining_inputs: CombiningInputs = combining_inputs
        self.output_weights: np.array([]) = np.array([])
        self.output_time: float = 0

    def calculate_weights_and_time(self):
        '''
        Get ILP weight vector result and runtime
        '''
        start_time = time.time()
        alpha_hat = combine_representations(self)
        elapsed_time = time.time() - start_time
        return {self.hash:(alpha_hat, elapsed_time)}


