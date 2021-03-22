"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import getopt

import logging
import os
import sys
import time

# standard numerical imports
import numpy as np
from colorama import Fore
# parallelization 
from joblib import Parallel, delayed
# progress bar import
from tqdm import tqdm

# python files imports
from ilp_common import *
from ilp_pipeline_config import *
from ilp_evaluation_proc import *
from ilp_pre_proc import *
from ilp_metadata_proc import *
from ilp_processing import *
from ilp_visualization import *
from ilp_manual_input import *
from ilp_common_classes import *

from typing import *
from typing import  List, Dict

def create_pipeline_config() -> ILPPipelineConfig:
    '''
    Receive Input and construct testing parameters
    '''
    #create a ILPPipelineConfig object
    pipeline_config                 = ILPPipelineConfig()

    #define command line parameter objects
    help_obj                        = ILPPipelineParameter('help',pipeline_config.help_handler)
    ifile_obj                       = ILPPipelineParameter('ifile', pipeline_config.ifile_handler, '../json/pipeline_driver_input_template.json')
    runid_obj                       = ILPPipelineParameter('runid',pipeline_config.runid_handler,'100')
    dataset_obj                     = ILPPipelineParameter('dataset', pipeline_config.dataset_handler, 'test_set_data')
    root_path_obj                   = ILPPipelineParameter('path_to_root',pipeline_config.path_to_root_handler,'../../')
    metadata_path_obj               = ILPPipelineParameter('path_to_metadata', pipeline_config.path_to_metadata_handler, '../data/test_set_data/metadata.json')
    output_path_obj                 = ILPPipelineParameter('path_to_output', pipeline_config.path_to_output_handler, '../output_data/output_test_set/')
    num_vois_obj                    = ILPPipelineParameter('num_vois',pipeline_config.num_vois_handler,'5')
    training_sets_sizes_obj         = ILPPipelineParameter('training_sets_sizes', pipeline_config.training_sizes_handler, '[10,15]')
    minimum_ranking_thresholds_obj  = ILPPipelineParameter('minimum_ranking_thresholds',pipeline_config.threshold_handler,'[None]')
    solvers_apis_obj                = ILPPipelineParameter('solvers_and_apis', pipeline_config.solver_api_handler, "[[pulp,coin_cmd]]")
    weight_approx_resolutions_obj   = ILPPipelineParameter('weight_approx_resolutions',pipeline_config.weight_approx_resolution_handler,'[None]')
    num_cores_obj                   = ILPPipelineParameter('num_cores',pipeline_config.num_cores_handler,'3')
    persist_data_obj                = ILPPipelineParameter('persist_data',pipeline_config.persist_data_handler,'True')
    mode_obj                        = ILPPipelineParameter('mode',pipeline_config.mode_handler,'DEBUG')
    eval_data_mode_obj              = ILPPipelineParameter('eval_data_mode',pipeline_config.eval_data_mode_handler,'RANDOM')
    training_data_mode_obj          = ILPPipelineParameter('training_data_mode',pipeline_config.training_data_mode_handler,'RANDOM')
    gurobi_outputflag_obj           = ILPPipelineParameter('gurobi_outputflag',pipeline_config.gurobi_outputflag_handler,'1')
    time_limit_obj                  = ILPPipelineParameter('time_limit',pipeline_config.time_limit_handler,'120')
    num_threads_obj                 = ILPPipelineParameter('num_threads',pipeline_config.num_threads_handler,'1')
    eval_size_obj                   = ILPPipelineParameter('eval_size',pipeline_config.eval_size_handler,'10')
    eval_param_obj = ILPPipelineParameter('eval_param', pipeline_config.eval_param_handler, 'training_sets_size')
    create_obj                      = ILPPipelineParameter('create',pipeline_config.create_handler,'foo.json')
    manual_obj                      = ILPPipelineParameter('manual',pipeline_config.manual_input_handler)
    indices_obj                     = ILPPipelineParameter('indices',pipeline_config.indices_handler, '[]')

    cmd_options = {
                '-h'                            : help_obj,
                '--help'                        : help_obj,
                '-i'                            : ifile_obj,
                '--ifile'                       : ifile_obj,
                '--runid'                       : runid_obj,
                '--dataset'                     : dataset_obj,
                '--root-path'                   : root_path_obj,
                '--metadata-path'               : metadata_path_obj,
                '--output-path'                 : output_path_obj,
                '--num-vois'                    : num_vois_obj,
                '--training-sets-sizes'         : training_sets_sizes_obj,
                '--minimum-ranking-thresholds'  : minimum_ranking_thresholds_obj,
                '--solvers-apis'                : solvers_apis_obj,
                '--weight-approx-resolutions'   : weight_approx_resolutions_obj,
                '--num-cores'                   : num_cores_obj,
                '--persist-data'                : persist_data_obj,
                '--mode'                        : mode_obj,
                '--eval-data-mode'              : eval_data_mode_obj,
                '--training-data-mode'          : training_data_mode_obj,
                '--gurobi-outputflag'           : gurobi_outputflag_obj,
                '--time-limit'                  : time_limit_obj,
                '--num-threads'                 : num_threads_obj,
                '--eval-size'                   : eval_size_obj,
                '--eval-param'                  : eval_param_obj,
                '-c'                            : create_obj,
                '--create'                      : create_obj,
                '--manual'                      : manual_obj,
                '--indices'                     : indices_obj
            }
    #Creating ILPParams Object
    ilp_params = ILPParams()

    #Assign values to pipeline_config object
    pipeline_config.ilp_params = ilp_params
    pipeline_config.cmd_options = cmd_options

    #receive non-default testing parameters from command line with method options
    argv = sys.argv[1:] 
    try:
        opts, params = getopt.getopt(argv,"hc:i:",["help","manual","runid=","create=","ifile=", "indices=", "dataset=",
                                        "root-path=","metadata-path=", "output-path=", "num-vois=","training-sets-sizes=",
                                        "minimum-ranking-thresholds=","solvers-apis=","weight-approx-resolutions=","num-cores=","persist-data=","mode=","eval-data-mode=","training-data-mode=","gurobi-outputflag=","time-limit=","num-threads=","eval-size=", "eval-param="])
    except getopt.GetoptError as e:
        raise ILPError(e)
        
    for opt, arg in opts:
        if opt not in pipeline_config.cmd_options.keys():
            raise ILPError(('{s} option not handeled!'.foramt(s=opt)))
        else:
            pipeline_config.cmd_update(opt,arg)

    return pipeline_config

def create_pre_proc_data(pipeline_config: ILPPipelineConfig) -> ILPPreProcData:
    '''
    Get pre-processed data
    '''
    all_data = ilp_metadata_json_parser()
    all_data.load_data(pipeline_config.ilp_params.path_to_metadata)

    pre_processed_data = ilp_pre_proc(pipeline_config.ilp_params, all_data, pipeline_config.voi_indices)

    #update the sizes of fixed eval and training data
    if pipeline_config.ilp_params.eval_data_mode == DataInputMode.FIXED:
        pipeline_config.ilp_params.eval_size = len(list(pre_processed_data.eval_dict.values())[0])
    if pipeline_config.ilp_params.training_data_mode == DataInputMode.FIXED:
        pipeline_config.ilp_params.training_sets_sizes = [len(list(pre_processed_data.training_dict.values())[0])]

    return pre_processed_data

def create_sub_dirs(pipeline_config: ILPPipelineConfig, api:str) -> tuple([str,ILPPipelineConfig]):

    output_sub_dir_path = pipeline_config.run_output_dir_path+'/' + pipeline_config.run_id+"_"+ str(pipeline_config.sub_run_id) + "_" + api
    create_dir(output_sub_dir_path, False)
    pipeline_config.sub_run_id += 1

    if output_sub_dir_path not in pipeline_config.experiment_dicts.keys(): #create empty lists for experiment_dicts only on first full loop
        pipeline_config.experiment_dicts[output_sub_dir_path] = []

    return (output_sub_dir_path, pipeline_config)

def get_training_data_for_v_star(v_star: int, pipeline_config: ILPPipelineConfig, pre_processed_data: ILPPreProcData, training_similar_sets_dict: Dict[str,List[int]],
                        training_set_size: int) -> tuple([CombiningInputs, Dict[str,Dict[str,List[int]]]]):
    #generate random selection:
    np.random.seed()
    
    if pipeline_config.ilp_params.training_data_mode == DataInputMode.FIXED:
        combining_inputs = get_similar_and_other_nodes(np.asarray(
            pre_processed_data.dist_matrices[str(v_star)]), v_star, pre_processed_data.training_dict[str(v_star)], pipeline_config.ilp_params)
        training_similar_sets_dict[str(training_set_size)] = pre_processed_data.training_dict[str(v_star)]

    elif pipeline_config.ilp_params.training_data_mode == DataInputMode.RANDOM:
        # create a similar_indicies set(in the same regions as v_stars) of size training_set_size
        try:
            training_similar_set_indices = np.random.choice([int(s) for s in pre_processed_data.testing_similar_sets_indices[str(v_star)] if (s not in pre_processed_data.eval_dict[str(v_star)] and not np.isnan(s))], training_set_size, replace=False)
        except ValueError as e:
                raise Exception("Choose smaller training_set_sizes, they're too high, allowing for an overlap between training and evaluation data. See manual for more details.") from e 
        #Grab a new CombiningInputs object for v* with the same training_set_size.
        combining_inputs = get_similar_and_other_nodes(np.asarray(pre_processed_data.dist_matrices[str(v_star)]), v_star, training_similar_set_indices, pipeline_config.ilp_params)
        training_similar_sets_dict[str(training_set_size)] = training_similar_set_indices.tolist()
    
    else:
        raise ILPError("training_data_mode is not supported, please use FIXED or RANDOM, see manual for more details.")

    return (combining_inputs, training_similar_sets_dict)

def setup_experiments(pipeline_config: ILPPipelineConfig, pre_processed_data: ILPPreProcData):
    '''
    Setup Experiments, organize testing parameters for ILP
    '''
    for vi, v_star in enumerate(pre_processed_data.v_stars):  
        v_star = int(v_star)
        training_similar_sets_dict = {}

        for i, training_set_size in enumerate(pipeline_config.ilp_params.training_sets_sizes):
            combining_inputs, training_similar_sets_dict = get_training_data_for_v_star(v_star, pipeline_config, pre_processed_data, training_similar_sets_dict, training_set_size)
            
            for ii, minimum_ranking_threshold in enumerate(pipeline_config.ilp_params.minimum_ranking_thresholds):
                for iii, solver_and_api in enumerate(pipeline_config.ilp_params.solvers_and_apis):
                    for iv, weight_approx_resolution in enumerate(pipeline_config.ilp_params.weight_approx_resolutions):

                        output_sub_dir_path, pipeline_config = create_sub_dirs(pipeline_config, solver_and_api[0])
                        #create a partial copy of CombiningInputs object to avoid altering the logger path of previous instances
                        combining_inputs_cpy = combining_inputs.create_copy(os.path.join(output_sub_dir_path,"gurobi_debug.log"))
                        voi_experiment = ILPVoiExperiment(v_star, training_set_size, solver_and_api[0], solver_and_api[1], minimum_ranking_threshold, weight_approx_resolution, combining_inputs_cpy)
                        pipeline_config.experiment_dicts[output_sub_dir_path].append(voi_experiment)

        pipeline_config.sub_run_id = 1
        pipeline_config.all_training_similar_sets_dict[str(v_star)] = training_similar_sets_dict
    return pipeline_config

def conduct_experiments(pipeline_config: ILPPipelineConfig):
    '''
    Conducte experiments in parallel
    '''
    condensed_func = lambda y: y.calculate_weights_and_time() 

    #perform experiments in parallel using joblib
    voi_exp_results_list = Parallel(n_jobs=pipeline_config.ilp_params.num_cores)(delayed(condensed_func)(y) for x in tqdm(pipeline_config.experiment_dicts.values(), bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)) for y in x)
    
    voi_exp_results_dict = convert_list_of_dicts_to_dict(voi_exp_results_list)

    for experiment in pipeline_config.experiment_dicts.values():
        for voi_obj in experiment:
            output = voi_exp_results_dict[voi_obj.hash]
            voi_obj.output_weights = output[0] 
            voi_obj.output_time = output[1]

def write_results(pipeline_config: ILPPipelineConfig, elapsed_time: float):
    '''
    Write ouput to output dirs/sub-dirs and files.
    '''
    for sub_dir_path, experiments_list in pipeline_config.experiment_dicts.items():
        total_time = 0
        weights_data = {}
        runtime_data = {}

        #create testing prameters JSON:
        test_data = {
            "num_vois": pipeline_config.ilp_params.num_vois,
            "training_sets_size": experiments_list[0].training_sets_size,
            "minimum_ranking_threshold": experiments_list[0].mrr,
            "api": experiments_list[0].api,
            "solver": experiments_list[0].solver,
            "weight_approx_resolution": experiments_list[0].war
        }

        for voi_experiment in experiments_list:
            weights_data[str(voi_experiment.v_star)] = voi_experiment.output_weights.tolist()
            runtime_data[str(voi_experiment.v_star)] = (float(voi_experiment.output_time))/60
            total_time+= float(voi_experiment.output_time)

        dump_json(test_data, os.path.join(sub_dir_path, "ilp_params.json"))
        dump_json(weights_data, os.path.join(sub_dir_path, "voi_weight_vectors.json"))
        dump_json(runtime_data, os.path.join(sub_dir_path, "voi_runtime.json"))

        if pipeline_config.ilp_params.mode == RunMode.DEBUG:
            with open(os.path.join(sub_dir_path, "gurobi_debug.log"), 'a') as log_file:
                log_file.write("\n\n--- Total time in minutes: %s for %s indices ---\n\n" % ((total_time/60), pipeline_config.ilp_params.num_vois))

        pipeline_config.logger.info("All weight vectors:")
        for voi, weights in weights_data.items():
            pipeline_config.logger.info("Weight Vector for voi %s: %s\n" % (voi, weights))
            
    num_experiments = len(pipeline_config.experiment_dicts.keys())
    pipeline_config.logger.info("\n--- Total time in minutes: %s for %s indices, running %s experiments each ---\n" % ((elapsed_time/60), pipeline_config.ilp_params.num_vois, num_experiments))
    
def eval_visualize(pipeline_config: ILPPipelineConfig, eval_dict: Dict[str,int]) -> bool:
    '''
    Evaluate and visualize results
    '''
    #evaluate the results relative to eval_param:'
    eval_tuple = eval_proc(pipeline_config.ilp_params.eval_param,pipeline_config.run_output_dir_path, eval_dict)
    logging.info('using evalulation parameter: %s', pipeline_config.ilp_params.eval_param.value)

    pipeline_config.logger.propagate = False

    if pipeline_config.ilp_params.eval_param == EvalParam.WAR:
        x_axis = pipeline_config.ilp_params.weight_approx_resolutions
    elif pipeline_config.ilp_params.eval_param == EvalParam.TRAINING_SIZE:
        x_axis = pipeline_config.ilp_params.training_sets_sizes

    visulaize_results(pipeline_config.run_output_dir_path, np.array(x_axis), eval_tuple[0],'mrr_values', pipeline_config.ilp_params.eval_param)

    visulaize_results(pipeline_config.run_output_dir_path, np.array(x_axis), eval_tuple[1],'runtime_values',pipeline_config.ilp_params.eval_param)

    return eval_tuple[2]

def handle_run_mode_ouput(pipeline_config: ILPPipelineConfig, pre_processed_data: ILPPreProcData):
    '''
    Handle input, evaluation and  visualization data based on run mode.
    '''
    #store all_ilp_params:
    with open(os.path.join(pipeline_config.run_output_dir_path, str(pipeline_config.run_id)+'_pipeline_ilp_params.json'),'w') as f:
        f.write(pipeline_config.ilp_params.__repr__())
    
    if pipeline_config.ilp_params.mode == RunMode.DEBUG:

        #dump similar nodes and distances data
        dump_json(pipeline_config.all_training_similar_sets_dict, os.path.join(pipeline_config.run_output_dir_path, 'voi_similar_indices.json'))
        dump_json(pre_processed_data.dist_matrices, os.path.join(pipeline_config.run_output_dir_path,'voi_distances.json'))

        #call the evaluation and visualization step
        pipeline_config.test_successful = eval_visualize(pipeline_config, pre_processed_data.eval_dict)
    
    return pipeline_config

def main():

    #set up
    pipeline_config = create_pipeline_config()
    pre_processed_data = create_pre_proc_data(pipeline_config)
    pipeline_config.create_output_dirs()
    pipeline_config.save_log_preproc_data_debug_mode(pre_processed_data)
    pipeline_config = setup_experiments(pipeline_config, pre_processed_data)
    
    #record total time for all experiments
    start_time = time.time()

    #conduct experiments in parallel
    conduct_experiments(pipeline_config)

    #save output
    elapsed_time = time.time() - start_time
    write_results(pipeline_config, elapsed_time)
    pipeline_config = handle_run_mode_ouput(pipeline_config, pre_processed_data)
    return pipeline_config.test_successful
    
main()
