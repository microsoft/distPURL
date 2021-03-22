"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import logging
import os
import sys

from typing import *
from typing import List, Any, Dict
from multiprocessing import Lock
from datetime import datetime
from pytz import timezone

from ilp_common import *
from ilp_common_classes import *
from ilp_manual_input import *
from ilp_voi_experiment import *
from ilp_pre_proc import *

class ILPPipelineParameter:
    def __init__(self, name: str, function_handler: Callable[[str],Any], value: Any = None, default: Any = None):
        self.name: str = name
        self.function_handler: Callable[[str],Any] = function_handler
        self.value: Any = value
        self.default: Any = default

class ILPPipelineConfig:
    def __init__(self):
        self.ilp_params: ILPParams
        self.cmd_options: Dict[str, ILPPipelineParameter]
        self.voi_indices: List[int] = []
        self.pipeline_driver_input_path: str = "../json/pipeline_driver_input_template.json" 
        self.run_id_input_flag: bool = False
        self.run_id_dir_path = '../../.ilp_code/'
        self.run_id: str = '100'
        self.sub_run_id: int = 1
        self.run_output_dir_path: str = ''
        self.experiment_dicts: Dict[str,List[ILPVoiExperiment]] = {}
        self.output_dir_paths: List[str] = []
        self.all_training_similar_sets_dict: Dict[strDict[str,int]] = {}
        self.logger: str = ''
        self.test_successful: bool = False 

    def reset_run_id(self):
        #increment run_id
        with open(os.path.join(self.run_id_dir_path, 'run_id_file.txt'), 'w') as run_id_file:
            run_id_file.write(str(int(self.run_id) + 1))
            
    def create_output_dirs(self):
        '''
        Create an output directory
        '''
        create_dir((self.ilp_params.path_to_root+'/src/output_data'),False)

        create_dir(self.ilp_params.path_to_output,False)

        if self.run_id_input_flag == False:
            get_run_id_lock = Lock()
            create_dir(self.run_id_dir_path, False)
            runid_file_path = os.path.join(self.run_id_dir_path, 'run_id_file.txt')
            if os.path.exists(runid_file_path) == False:
                with open(runid_file_path, 'w') as f:
                    f.write(str(self.run_id))
            with get_run_id_lock:
                with open(runid_file_path, 'r') as run_id_file:
                    self.run_id = run_id_file.read()
                    self.reset_run_id()
                    
        self.run_output_dir_path = os.path.join(self.ilp_params.path_to_output, '{d}_{r}_{t}'.format(d=self.ilp_params.dataset, r=self.run_id, t=(datetime.now(timezone('US/Pacific')).strftime('_%m%d%Y_%H:%M:%S_%z'))))

        create_dir(self.run_output_dir_path, True)

    def save_log_preproc_data_debug_mode(self, pre_processed_data:ILPPreProcData):
        '''
        Configure logger
        '''
        self.logger = logging.getLogger('ilp_pipeline_driver')
        self.logger.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        '''
        Save logging to file in debug mode
        '''
        if self.ilp_params.mode == RunMode.DEBUG:
            # create file handler and set level to debug
            fh = logging.FileHandler(os.path.join(self.run_output_dir_path,'ilp_pipeline_logging.log'),'w')
            fh.setLevel(logging.DEBUG)

            # add formatter to fh
            fh.setFormatter(formatter)

            # add fh to logger
            self.logger.addHandler(fh)
        
        '''
        Save preprocessed data in debug mode
        '''
        if self.ilp_params.persist_data == True and self.ilp_params.mode == RunMode.DEBUG:
            try:
                dump_json(pre_processed_data.__dict__, os.path.join(self.run_output_dir_path,'pre_processed_data.json'))
            except ValueError:
                raise ILPError('Output directory needs to be created before saving pre-processed data.')
            self.logger.info("pre-processed data loaded to JSON")

    def help_handler(self, arg: List[str]):
        print('\n-- use default values for all testing parameters --')
        print('ilp_pipeline_driver.py\n')
        print("-- modify specific parameters --")
        print("ilp_pipeline_driver.py --dataset --root-path --metadata-path --output-path --num-vois --training-sets-sizes --minimum-ranking-thresholds --solvers-apis --weight-approx-resolutions --num-cores --persist-data --mode --eval-data-mode --training-data-mode --gurobi-outputflag --time-limit --num-threads --eval-size --eval-param\n")
        print('-- create a new json input file using current testing parameters: [-c or --create] (avoid spaces in list params) --')
        print('ilp_pipeline_driver.py -c <inputfile.json>\n')
        print('-- use a pre-written json input file: [-i or --ifile] --')
        print('ilp_pipeline_driver.py -i <inputfile.json>\n')
        print('-- enter parameters one by one (for users unfamiliar with required input): [--manual] --')
        print('ilp_pipeline_driver.py --manual\n')
        print('-- specify your run label: [--runid] --')
        print('ilp_pipeline_driver.py --runid <run_label>\n')
        print('-- specify your vertices of interest: [--indices] --')
        print('ilp_pipeline_driver.py --indices <[voi1,voi2,..,voi_n]>\n')
        sys.exit(0)

    def ifile_handler(self, arg: List[str]):
        ilp_params = ILPParams()
        ilp_params.load_data(arg[0])
        self.ilp_params = ilp_params
        self.pipeline_driver_input_path = arg[0]

    def runid_handler(self, arg: List[str]):
        self.run_id_input_flag = True
        self.run_id = arg[0]

    def dataset_handler(self, arg: List[str]):
        self.ilp_params.dataset = arg[0]

    def path_to_root_handler(self, arg: List[str]):
        self.ilp_params.path_to_root = arg[0]

    def path_to_metadata_handler(self, arg: List[str]):
        self.ilp_params.path_to_metadata = arg[0]

    def path_to_output_handler(self, arg: List[str]):
        self.ilp_params.path_to_output = arg[0]

    def num_vois_handler(self, arg: List[str]):
        self.ilp_params.num_vois = int(arg[0])

    def training_sizes_handler(self, arg: List[str]):
        self.ilp_params.training_sets_sizes = [int(i) for i in arg[0].strip().strip('[]').split(',')]

    def threshold_handler(self, arg: List[str]):
        self.ilp_params.minimum_ranking_thresholds = [int(i) if (i != 'None') else None for i in arg[0].strip().strip('[]').split(',')]

    def solver_api_handler(self, arg: List[str]):
        self.ilp_params.solvers_and_apis = [[i for i in j.split(',')] for j in arg[0].strip().strip('[]').split('],[')]

    def weight_approx_resolution_handler(self, arg: List[str]):
        self.ilp_params.weight_approx_resolutions = [float(i) if (i != 'None') else None for i in arg[0].strip().strip('[]').split(',')]

    def num_cores_handler(self, arg: List[str]):
        self.ilp_params.num_cores = int(arg[0])

    def persist_data_handler(self, arg: List[str]):
        self.ilp_params.persist_data = True if arg[0].lower()=='true' else False

    def mode_handler(self, arg: List[str]):
        self.ilp_params.mode = RunMode(arg[0].upper())

    def eval_data_mode_handler(self, arg: List[str]):
        self.ilp_params.eval_data_mode = DataInputMode(arg[0].upper())

    def training_data_mode_handler(self, arg: List[str]):
        self.ilp_params.training_data_mode = DataInputMode(arg[0].upper())

    def gurobi_outputflag_handler(self, arg: List[str]):
        self.ilp_params.gurobi_outputflag = int(arg[0])

    def time_limit_handler(self, arg: List[str]):
        self.ilp_params.time_limit = float(arg[0])

    def num_threads_handler(self, arg: List[str]):
        self.ilp_params.num_threads = int(arg[0])

    def eval_size_handler(self, arg: List[str]):
        self.ilp_params.eval_size = int(arg[0])

    def eval_param_handler(self, arg: List[str]):
        self.ilp_params.eval_param = EvalParam(arg[0])

    def create_handler(self, arg: List[str]): 
        drive_archive_dir = os.path.join(self.ilp_params.path_to_root, 'src/driver_input_temp')
        create_dir(drive_archive_dir,False)
        self.pipeline_driver_input_path = os.path.join(drive_archive_dir, arg[0])
        with open(self.pipeline_driver_input_path, 'w') as json_file:
            json_file.write(self.ilp_params.__repr__())

    def manual_input_handler(self, arg: List[str]):
        ilp_params = ILPParams()
        pipeline_driver_input_path = input_from_cmd(ilp_params)
        self.ilp_params = ilp_params
        self.pipeline_driver_input_path = pipeline_driver_input_path

    def indices_handler(self, arg: List[str]):
        voi_indices = [int(i) for i in arg[0].strip().strip('[]').split(',')]
        self.voi_indices = voi_indices
        self.ilp_params.num_vois = len(voi_indices)

    def cmd_update(self,opt:str,arg: str):
        self.cmd_options[opt].function_handler([arg])
