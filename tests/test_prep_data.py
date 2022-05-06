import pytest
import sys
import os 
import shutil
sys.path.append('code')
from prep_data_functions import *


@pytest.mark.download_dataset
def test_download_dataset(tmp_path):
    data_dir = tmp_path
    dataset_name = 'spaceship-titanic' 

    download_dataset(data_dir, dataset_name)
    data_set_path = os.path.join(data_dir,dataset_name)

    assert os.path.isdir(data_set_path) == True



@pytest.mark.work_space
def test_create_work_space(tmp_path):
    data_dir = tmp_path
    created_paths = []

    # Add directories that should be created to created_paths
    train_dir = os.path.join(data_dir, "tf_record_data")
    created_paths.append(train_dir)
    create_dirs = ["annotations", "images", "training_data"]
    for dir in create_dirs:
        created_paths.append(os.path.join(train_dir, dir))
    
    create_work_space(data_dir)

    for dir in created_paths:
        assert os.path.isdir(dir) == True
    

