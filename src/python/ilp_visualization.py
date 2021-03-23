"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import *
from typing import List

from ilp_common_classes import *

logger = logging.getLogger('matplotlib')
logger.setLevel(logging.WARNING)

def visulaize_results(path_to_output_data: str, var_param: List[float], results: np.array, results_name: str, var_name: EvalParam):
    '''
    Plots figures representing the evaluation results of the ILP pipeline.
    '''
    fig, ax = plt.subplots(1,1, figsize=(8,4))

    #replace None weight_approx_resolution values with 0
    if None in var_param:
        var_param = np.array(var_param)
        var_param = np.where(var_param == None, 0.0, var_param)
    
    #plot evaluation results
    ax.plot(var_param, np.mean(results, axis=-1), linestyle='--', marker='o')

    #customize figure
    plt.title(str(len(var_param))+' '+var_name.value+' vs. '+str(results_name)+' for '+str(len(results[0]))+ ' vois', y=1.08)
    plt.xlabel(var_name.value, color = 'red')
    plt.ylabel(str(results_name), color='red')
    plt.xticks(var_param.astype(np.double), color='blue')
    plt.grid(True)

    #save plot to output path
    fig.savefig(os.path.join(path_to_output_data, results_name+'_vs_'+var_name.value+'_plot.png'), bbox_inches='tight',dpi=300)
    plt.close(fig)
