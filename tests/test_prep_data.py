import pytest
import sys
import os
import pandas as pd
sys.path.append("code")
from prep_data_functions import *


@pytest.mark.download_dataset
def test_download_dataset(tmp_path):
    data_dir = tmp_path
    dataset_name = "spaceship-titanic"

    download_dataset(data_dir, dataset_name)
    data_set_path = os.path.join(data_dir, dataset_name)

    assert os.path.isdir(data_set_path) == True


@pytest.mark.workspace
def test_create_workspace(tmp_path):
    data_dir = tmp_path
    created_paths = []

    # Add directories that should be created to created_paths
    train_dir = os.path.join(data_dir, "tf_record_data")
    created_paths.append(train_dir)
    create_dirs = ["annotations", "images", "training_data"]
    for dir in create_dirs:
        created_paths.append(os.path.join(train_dir, dir))

    create_workspace(data_dir)

    for dir in created_paths:
        assert os.path.isdir(dir) == True

@pytest.mark.fix_annotation
def test_fix_anno():
    raw_annotation = "[{'x': 518, 'y': 165, 'width': 73, 'height': 56}]"
    fixed_annotation = [{'x': 518, 'y': 165, 'width': 73, 'height': 56}]

    result = fix_anno(raw_annotation)
    assert fixed_annotation == result

@pytest.mark.to_pascal
def test_to_pascal():
    data = {'annotations_coco':[{'x': 559, 'y': 213, 'width': 50, 'height': 32}]}
    row = pd.Series(data)
    pascal_annotation = [{'x_left': 559, 'y_top': 213, 'x_right': 609, 'y_bottom': 245}]

    result = to_pascal(row)
    assert result == pascal_annotation

@pytest.mark.box_id
def test_box_id():
    class_id = 1
    data = {'annotations_pascal':[{'x_left': 559, 'y_top': 213, 'x_right': 609, 'y_bottom': 245}, {'x_left': 559, 'y_top': 213, 'x_right': 609, 'y_bottom': 245}]}
    row = pd.Series(data)
    valid_output = [1,1]

    result = box_id(row, class_id)
    assert result == valid_output

@pytest.mark.box_class_names
def test_box_class_names():
    class_name = 'cots'
    data = {'annotations_pascal':[{'x_left': 559, 'y_top': 213, 'x_right': 609, 'y_bottom': 245}, {'x_left': 559, 'y_top': 213, 'x_right': 609, 'y_bottom': 245}]}
    row = pd.Series(data)
    valid_output = ['cots','cots']

    result = box_class_names(row, class_name)
    assert result == valid_output


@pytest.mark.create_label_map
def test_create_label_map(tmp_path):
    data_dir = tmp_path
    class_id = 1
    class_name = 'cots'
    label_map = f"""item {{
    id:{class_id}
    name:"{class_name}"
    }} """

    os.mkdir(os.path.join(data_dir,"tf_record_data"))
    os.mkdir(os.path.join(data_dir, "tf_record_data", "training_data"))

    create_label_map(data_dir, class_id, class_name)

    path = os.path.join(data_dir,"tf_record_data", "training_data","label_map.txt")
    with open(path) as f:
        result = f.readlines()

    result = ''.join(line for line in result)

    assert result == label_map
    
@pytest.mark.split_train_val
def test_split_train_val():
    data = {'c1':[1,2,8,5,8,1,4,5,6,7],'c2':[5,2,3,5,6,6,7,8,1,3]}
    df = pd.DataFrame(data=data)
    ratio = 0.9
    n_train_records = 9
    n_valid_records = 1

    train_df, valid_df  = split_train_val(train_meta_df=df, ratio=ratio)

    n_result_train = train_df.shape[0]
    n_result_valid = valid_df.shape[0]

    assert n_result_train == n_train_records and n_result_valid == n_result_valid


@pytest.mark.create_tf_example
def test_create_tf_example():
    row = train_meta_df.iloc[16]
    k = create_tf_example(row, data_path=data_dir)
    for key, value in k.features.feature.items():
        print(key)


