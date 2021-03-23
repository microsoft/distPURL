"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import click

from typing import *

from ilp_common_classes import *
from ilp_common import *

def input_from_cmd(params: object) -> str:
    '''
    Collects input from the user when they run the pipeline using the --manual option and returns the name of the json file the user just created to hold all their selected testing parameters.
    '''
    params.path_to_root = click.prompt("What's your path to root",default='../../')
    params.dataset = click.prompt("What's the name of the dataset you're using")
    json_file_name = click.prompt("What would you like to call your testing parameters JSON file")
    params.path_to_metadata = click.prompt("Please enter path to metadata for your dataset")
    params.path_to_output = click.prompt("Please enter path to write output to")
    params.persist_data = bool(click.prompt("Do you want to persist pre-processed data to memory [True/False]"))
    params.mode = RunMode((click.prompt("What mode would you like to use [DEBUG/RELEASE]")).upper())
    params.eval_data_mode = DataInputMode(click.prompt("What evaluation input data mode will you use [RANDOM/FIXED]").upper())
    params.training_data_mode = DataInputMode(click.prompt("What training input data mode will you use [RANDOM/FIXED]").upper())

    print("\n-------------------What experiments do you want to run?----------------------\n")
    
    params.num_vois = int(click.prompt('Number of vertices you would like to test for', default=1))

    input_s = click.prompt('Training similar set sizes (press q when done)')
    params.training_sets_sizes = []
    while(input_s!='q'):
        params.training_sets_sizes.append(int(input_s))
        input_s = click.prompt('Training similar set sizes (press q when done)')

    input_t = click.prompt('Minimum ranking thresholds (press q when done)')
    params.minimum_ranking_thresholds = []
    while(input_t!='q'):
        if(input_t == 'None'):
            params.minimum_ranking_thresholds.append(None)
        else:
            params.minimum_ranking_thresholds.append(int(input_t))
        input_t = click.prompt('Minimum ranking thresholds (press q when done)')

    input_sa = click.prompt('Apis and solvers in the form of a tuple (api,solver) (press q when done)')
    params.solvers_and_apis = []
    while(input_sa!='q'):
        x = tuple(map(str, input_sa.strip('()').split(',')))
        params.solvers_and_apis.append(x)
        input_sa = click.prompt('Apis and solvers in the form of a tuple (api,solver) (press q when done)')

    input_v= click.prompt('Weight approximation resolutions (press q when done)')
    params.weight_approx_resolutions = []
    while(input_v!='q'):
        if(input_v == 'None'):
            params.weight_approx_resolutions.append(None)
        else:
            params.weight_approx_resolutions.append(float(input_v))
        input_v = click.prompt('Weight approximation resolutions (press q when done)')

    params.time_limit = click.prompt("Time limit for the optimizer api in seconds", default=120)
    params.num_cores = click.prompt("Number of cores to use", default= 32)
    params.num_threads = click.prompt("Number of threads for the optimizer api", default = 25)
    params.gurobi_outputflag = click.prompt("Do you want to receive logging output from optimizer api",default = 1)
    params.eval_size = click.prompt("Evaluation set size",default = 5)
    params.eval_param = EvalParam(click.prompt("Parameter to evaluate relative to", default = 'training_sets_size'))
    click.confirm('Do you want to continue?', abort=True)

    pipeline_driver_input_path = os.path.join(params.path_to_root, 'src/driver_input_temp',json_file_name)
    drive_archive_dir = os.path.join(params.path_to_root, 'src/driver_input_temp')
    create_dir(drive_archive_dir,False)
    with open(pipeline_driver_input_path,'w') as f:
        f.write(params.__repr__()) 
    return (pipeline_driver_input_path)
