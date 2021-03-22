"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""
import os
import shutil
import numpy as np
import json

from pathlib import Path
from typing import *
from typing import List, Any, Dict

def dump_json(data: Dict[str, Any], path: str):
    with open(path,'w') as f:
        f.write(json.dumps(data, indent=2))

def convert_list_of_dicts_to_dict(list_dicts: List[Dict[str,Any]]) -> Dict[str,Any]:
    combined_dict = {}
    for dict in list_dicts:
        combined_dict.update(dict)
    return combined_dict

def create_dir(path: str, force:bool) -> str:
    '''
    Creates a directory given a full path.
    '''
    ret:str = ''
    
    try:
        dir = Path(path)

        if str(path) == str(os.getcwd()):
            raise ILPError('trying to delete current directory')
        if force and dir.exists() and dir.is_dir():
            shutil.rmtree(dir)

        dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(e)
        print("Failed to create %s directory" % (dir))
    return 

def list_dirs(folder: str) -> List[str]:
    '''
    Lists all directories under a given directory path.
    '''
    return [
        os.path.abspath(d) for d in (os.path.join(folder, d1) for d1 in os.listdir(folder))
        if os.path.isdir(d)
    ]

def list_files(folder: str) -> List[str]:
    '''
    Lists all files under a given directory path.
    '''
    return[
        f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
    ]

def validate_node_type(arr: np.array, data_type: np.dtype) -> bool:
    arr_tmp = arr
    if arr.dtype == object:
        arr_tmp = np.concatenate(arr)
    if arr_tmp.dtype != data_type:
        return False
    else:
        return True
