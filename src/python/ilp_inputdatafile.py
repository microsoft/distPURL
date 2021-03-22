"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""

import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from ilp_common import *
from ilp_common_classes import *

from typing import *
from typing import  List, Union, Dict

class InputDataFile:
    '''
    InputDataFile class: contains all metadata attributes related to input data files.
    '''
    def __init__(self, data):
        self.file_pattern: str = None
        self.url: str = None
        self.path: str = None
        self.file_type: str = None
        self.input_format_type: str = None
        self.col_of_interest_indx: int = None
        self.header_row: int = None
        self.header: List[str] = []
        self.header_type: List[str] = []
        self.excluded: bool = False
        try:
            self.__dict__.update(data)
        except TypeError:
            self = None

    def read_url(self):
        if self.url != '' or self.url != None:
            logging.warning('URL will not be used, modify InputDataFile: read_url() to handle reading')

    # check if single files are excluded
    def check_file_data_exists(self) -> bool:
        if self.path is None:
            raise ILPError("Sufficient data hasn't been provided for {s}, you might need to set your file's 'excluded' option to false".format(s = self.path))
        elif self.excluded == True: #only check if not None 
            raise ILPError("Sufficient data hasn't been provided for {s}, you might need to set your file's 'excluded' option to false".format(s = self.path))

    def csv_reader(self) -> np.array:
        data = np.array(pd.read_csv(self.path, header=self.header_row))
        return data

    def pkl_reader(self) -> np.array:
        data = np.array(pickle.load(open(self.path, 'rb')))
        return data

    def npy_reader(self) -> np.array:
        data = np.load(self.path, allow_pickle=True)
        return data

    def json_reader(self) -> Dict:
        with open(self.path) as f:
            data = json.load(f)
        return data

    def read_file_to_dict(self,ignore_nan:bool, datatype:str='') -> Dict[str,List[int]]:
        if os.path.exists(self.path) == False:
            raise ILPError(
                "data file {s}, doesn't exist".format(s=self.path))
        file_type = FileExtension(self.file_type)
        if file_type == FileExtension.CSV:
            data = pd.read_csv(self.path, header=self.header_row)
            if ignore_nan:
                data = data.dropna()
            if datatype != '':
                data = data.astype(datatype)
            data = dict(data)
        elif file_type == FileExtension.JSON:
            data = self.json_reader()
        else:
            raise ILPError("your data file {s} is not in the correct format, please check the assisting manual for supported formats for different data files.".format(s = self.path))
        return data

    def read_file_to_array(self) -> np.array:
        if os.path.exists(self.path) == False:
            raise ILPError(
                "data file {s}, doesn't exist".format(s=self.path))
        try:
            file_type = FileExtension(self.file_type)
        except ValueError:
            raise ILPError("your data file {s} is not in the correct format, please check the assisting manual for supported formats for different data files.".format(s = self.path))
        if file_type == FileExtension.PKL:
            data = self.pkl_reader()
        elif file_type == FileExtension.CSV:
            data = self.csv_reader()
        elif file_type == FileExtension.NPY:
            data = self.npy_reader()
        return data

    def read_data(self) -> Union[np.array, Dict]:
        if os.path.exists(self.path) == False:
            raise ILPError(
                "data file {s}, doesn't exist".format(s=self.path))
        '''
        Reads input files of different input formats, see descriptions below:
        ''' 
        if self.input_format_type is None:
            raise ILPError(
                "Input Format type not supported, please choose ONE SET, HEADER VOIS or COLUMN VOIS, double check your data file: {s}".format(s=self.path))
        try:
            input_format = InputFormat(self.input_format_type.upper())
        except ValueError:
            raise ILPError("Input Format type not supported, please choose ONE SET, HEADER VOIS or COLUMN VOIS, double check your data file: {s}".format(s=self.path))

        if input_format == InputFormat.ONE_SET:
            '''
            The input is a csv/npy/pkl file that contains a list of vertices of interest that are all similar to each other, for example, a csv file can look like the following, where col_of_interest_indx = 0:

            voi_id,col_1,col_2
            1111,0,0
            2222,2,1
            3333,5,2
            .....
            '''
            data = self.read_file_to_array()
            if self.col_of_interest_indx != None:
                data = data[:, self.col_of_interest_indx]
            if not validate_node_type(data, int):
                raise ILPError('Nodes need to be integers, double check your data file: {s}'.format(s = self.path))
        elif input_format == InputFormat.HEADER_VOIS:
            '''
            The input is a csv/json file that contains a mapping between vertices of interest and corresponding similar nodes, for example, a csv file can look like the following, where header_row = 0, corresponding to the vertices of interest (vois) and columns correspond to the similar nodes:

            1111,2222,3333
            5555,6666,7777
            2222,5555,1111
            3333,8888,7777
            .....
            '''
            if self.header_row is None and self.file_type not in '.json':
                raise ILPError('You need to include headers corresponding to your vertices of interest or use a json file, double check your data file: {s}'.format(s = self.path))
            data = self.read_file_to_dict(True,'int64')
            #validate nodes' type
            if not validate_node_type(np.array(list(data.values())), int):
                raise ILPError('Nodes need to be integers, double check your data file: {s}'.format(s = self.path))

        elif input_format == InputFormat.COLUMN_VOIS:
            '''
            The input is a csv/pkl/npy file that contains a mapping between vertices of interest and corresponding similar nodes, for example, a csv file can look like the following, where col_of_interest_indx = 0, corresponding to the vertices of interest (vois) and rows corresponding to the similar nodes:

            1111,5555,9999
            2222,6666,7777
            3333,5555,1111
            4444,8888,7777
            .....
            '''
            data = {}
            if self.col_of_interest_indx is None and self.file_type not in '.json':
                raise ILPError('You need to include a column of interest corresponding to your vertices of interest or use json file, double check your data file: {s}'.format(s = self.path))
            data_array = self.read_file_to_array()

            nrows=data_array.shape[0]
            ncols=data_array.shape[1]
            
            for i, voi in enumerate(data_array[:nrows, :1]):
                data[str(int(voi))] = np.array([int(j) for j in data_array[i, 1:ncols] if not np.isnan(j)])

            #validate nodes' type
            if not validate_node_type(np.array(list(data.values()),dtype=object), int):
                raise ILPError('Nodes need to be integers, double check your data file: {s}'.format(s = self.path))
        return data
