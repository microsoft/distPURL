"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import json
import os
import numpy as np
from sklearn.metrics import pairwise_distances

from ilp_common import *
from ilp_common_classes import *
from ilp_inputdatafile import *

from typing import *
from typing import  List, Union, Dict
import glob

def read_multiple_files(file_obj_list: List[InputDataFile]) -> List[np.array]:
    data_list = []
    for file_obj in file_obj_list:
        if file_obj.excluded == True:
            pass
        if file_obj.file_pattern is None:
            # if there is no file pattern, then this isn't multiple files in a dir, it's just 
            # one data file and path is its path not a dir path.
            data_list.append(file_obj.read_file_to_array())
        else:  # path is for a dir that includes multiple data files with file pattern
            data_dir_path = file_obj.path
            for data_file in glob.glob(os.path.join(data_dir_path,file_obj.file_pattern)):
            # use only files that end with file pattern
                file_obj.path = data_file
                data_list.append(file_obj.read_file_to_array())
            file_obj.path = data_dir_path
    return data_list

class ilp_metadata_json_parser:
    '''
    ilp_metadata_json_parser class: organizes metadata relating to input data files.
    '''
    def __init__(self):
        self.voi_indices_file: InputDataFile
        self.embeddings_files: list[InputDataFile] = []
        self.distance_matrices_files: List[InputDataFile] = []
        self.testing_data_file: InputDataFile
        self.eval_data_file: InputDataFile
        self.training_data_file: InputDataFile

    def load_data(self, path: str):
        with open(path) as f:
            data = json.load(f)
            self.voi_indices_file   = InputDataFile(data['voi_indices_file'])
            self.testing_data_file  = InputDataFile(data['testing_data_file'])
            self.eval_data_file     = InputDataFile(data['eval_data_file'])
            self.training_data_file = InputDataFile(data['training_data_file'])
            for file in data['embeddings_files']:
                self.embeddings_files.append(InputDataFile(file))
            for file in data['dist_matrices_files']:
                self.distance_matrices_files.append(InputDataFile(file))

    def get_vois(self) -> np.array:
        self.voi_indices_file.read_url()
        self.voi_indices_file.check_file_data_exists()
        if self.voi_indices_file.col_of_interest_indx != None:
            voi_indices = ((self.voi_indices_file).read_file_to_array())[
                :, self.voi_indices_file.col_of_interest_indx]
        else:
            voi_indices = self.voi_indices_file.read_file_to_array()
        if not validate_node_type(voi_indices, int):
            raise ILPError('Vertices of interest need to be integers, double check your data file: {s}'
                            .format(s = self.voi_indices_file.path))
        return voi_indices

    def get_embeddings(self) -> np.array:
        '''
        Takes multiple embedding input files and combines them using the pairwise_distances function. 
        Rows correspond to embedding vectors for each node and each vector contains numeric values of 
        differrent unique features that represent the embedding.
        '''
        self.embeddings_files[0].read_url()
        voi_indices = self.get_vois()
        if self.embeddings_files[0].path is None:
            raise ILPError('you need to provide a distance_matrices_files or embeddings_files.')
        if len(self.embeddings_files) == 1:
            self.embeddings_files[0].check_file_data_exists()
        embeddings = read_multiple_files(self.embeddings_files)
        dist_matrices = np.array([pairwise_distances(embed) for embed in embeddings])[
            :, voi_indices, :].transpose(1, 2, 0)
        return dist_matrices

    def get_distance_matrices(self) -> np.array:
        self.distance_matrices_files[0].read_url()
        if len(self.distance_matrices_files) == 1:
            if self.distance_matrices_files[0].path is None or self.distance_matrices_files[0].excluded == True:
                dist_matrices = self.get_embeddings()
                if not validate_node_type(dist_matrices, float):
                    raise ILPError('Embedding values need to be floats, double check your data file: {s}'
                                    .format(s = self.embeddings_files[0].path))
                return dist_matrices
        dist_matrices = np.concatenate(
            read_multiple_files(self.distance_matrices_files))
        if not validate_node_type(dist_matrices, float):
            raise ILPError('Distance values need to be floats, double check your data file: {s}'
                            .format(s = self.distance_matrices_files[0].path))
        return dist_matrices

    def get_eval_set(self) -> Dict:
        self.eval_data_file.read_url()
        if self.eval_data_file.path is None or self.eval_data_file.excluded == True:
            self.testing_data_file.check_file_data_exists()
            return None
        eval_set = self.eval_data_file.read_data()
        return eval_set

    def get_training_set(self) -> Dict:
        self.training_data_file.read_url()
        if self.training_data_file.path is None or self.training_data_file.excluded == True:
            self.testing_data_file.check_file_data_exists()
            return None
        training_set = self.training_data_file.read_data()
        return training_set

    def get_testing_set(self) -> Union[Dict, np.array]:
        self.testing_data_file.read_url()
        if self.testing_data_file.path is None or self.testing_data_file.excluded == True:
            self.eval_data_file.check_file_data_exists()
            self.training_data_file.check_file_data_exists()
            return None
        testing_set = self.testing_data_file.read_data()
        return testing_set
