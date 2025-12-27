import os
import yaml
from src.FlaskMlops import logger
import json
import joblib
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
from box.exceptions import BoxValueError



@ensure_annotations
def read_yaml(path_to_yaml:Path)-> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            logger.info(f"Yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e
    
    
@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")
            
            

@ensure_annotations
def save_json(path:Path, data:dict):
    with open (path,"w") as f:
        json.dump(data,f,index=4)
    logger.info(f"Json file saved at: {path}")
    
    
@ensure_annotations
def load_json(path:Path)->ConfigBox:
    with open(path) as f:
        content=json.load(f)
    logger.info(f"Json file loaded successfully from {path}")
    return ConfigBox(content)
    
@ensure_annotations
def save_model(data:Any,path:Path):
    joblib.dump(value=data,filename=path)
    logger.info(f"Binary file saved at: {path}")
    

@ensure_annotations
def load_model(path:Path)->Any:
    data=joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data