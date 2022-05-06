import tensorflow as tf
import os
import shutil
import pandas as pd
from object_detection.utils import dataset_util
import json
from PIL import Image
import io
import numpy as np
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi




def download_dataset(data_dir, dataset_name):
    '''
        Download the dataset if not already exists
        Note -> Need to add your kaggle acc and api in kaggle.json file on computer for access to API
    '''
    competition_dir = os.path.join(data_dir,dataset_name)
    if not os.path.exists(os.path.join(data_dir,dataset_name)):
        api = KaggleApi()
        api.authenticate()
        print('Dataset is around 15 GB may take a long time...')
        api.competition_download_files(dataset_name, path=data_dir)
        
        os.mkdir(competition_dir)
        with zipfile.ZipFile(os.path.join(data_dir,f'{dataset_name}.zip')) as zip_file:
            zip_file.extractall(competition_dir)
        os.remove(os.path.join(data_dir,f'{dataset_name}.zip'))
        print('Download complete and fils unziped!')
    else:
        print('Dataset folder already present!')
    return competition_dir

def create_work_space(data_dir):
    '''
        Takes in the directory to create workspace in.
    '''
    if not os.path.exists(os.path.join(data_dir, "tf_record_data")):
        os.mkdir(os.path.join(data_dir, "tf_record_data"))

        train_dir = os.path.join(data_dir, "tf_record_data")
        create_dirs = ["annotations", "images", "training_data"]
        for dir in create_dirs:
            exists = os.path.exists(os.path.join(train_dir, dir))
            if not exists:
                os.mkdir(os.path.join(train_dir, dir))
        print('Work space created.')
    else:
        print('Work space already exists!')

