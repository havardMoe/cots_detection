import pytest
import sys
import os
import pandas as pd
from PIL import Image
import tensorflow as tf
import io
import numpy as np
sys.path.append("code")
from prep_data_functions import *


@pytest.mark.download_dataset
def test_download_dataset(tmp_path):
    data_dir = tmp_path
    dataset_name = "spaceship-titanic"

    download_dataset(data_dir, dataset_name)
    data_set_path = os.path.join(data_dir, dataset_name)

    assert os.path.isdir(data_set_path)


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
        assert os.path.isdir(dir)


@pytest.mark.fix_annotation
def test_fix_anno():
    raw_annotation = "[{'x': 518, 'y': 165, 'width': 73, 'height': 56}]"
    fixed_annotation = [{"x": 518, "y": 165, "width": 73, "height": 56}]

    result = fix_anno(raw_annotation)
    assert fixed_annotation == result


@pytest.mark.to_pascal
def test_to_pascal():
    data = {
        "annotations_coco": [{"x": 559, "y": 213, "width": 50, "height": 32}]
    }
    row = pd.Series(data)
    pascal_annotation = [
        {"x_left": 559, "y_top": 213, "x_right": 609, "y_bottom": 245}
    ]

    result = to_pascal(row)
    assert result == pascal_annotation


@pytest.mark.box_id
def test_box_id():
    class_id = 1
    data = {
        "annotations_pascal": [
            {"x_left": 559, "y_top": 213, "x_right": 609, "y_bottom": 245},
            {"x_left": 559, "y_top": 213, "x_right": 609, "y_bottom": 245},
        ]
    }
    row = pd.Series(data)
    valid_output = [1, 1]

    result = box_id(row, class_id)
    assert result == valid_output


@pytest.mark.box_class_names
def test_box_class_names():
    class_name = "cots"
    data = {
        "annotations_pascal": [
            {"x_left": 559, "y_top": 213, "x_right": 609, "y_bottom": 245},
            {"x_left": 559, "y_top": 213, "x_right": 609, "y_bottom": 245},
        ]
    }
    row = pd.Series(data)
    valid_output = ["cots", "cots"]

    result = box_class_names(row, class_name)
    assert result == valid_output


@pytest.mark.create_label_map
def test_create_label_map(tmp_path):
    data_dir = tmp_path
    class_id = 1
    class_name = "cots"
    label_map = f"""item {{
    id:{class_id}
    name:"{class_name}"
    }} """

    os.mkdir(os.path.join(data_dir, "tf_record_data"))
    os.mkdir(os.path.join(data_dir, "tf_record_data", "training_data"))

    create_label_map(data_dir, class_id, class_name)

    path = os.path.join(
        data_dir, "tf_record_data", "training_data", "label_map.txt"
    )
    with open(path) as f:
        result = f.readlines()

    result = "".join(line for line in result)

    assert result == label_map


@pytest.mark.split_train_val
def test_split_train_val():
    data = {
        "c1": [1, 2, 8, 5, 8, 1, 4, 5, 6, 7],
        "c2": [5, 2, 3, 5, 6, 6, 7, 8, 1, 3],
    }
    df = pd.DataFrame(data=data)
    ratio = 0.9
    n_train_records = 9
    n_valid_records = 1

    train_df, valid_df = split_train_val(train_meta_df=df, ratio=ratio)

    n_result_train = train_df.shape[0]
    n_result_valid = valid_df.shape[0]

    assert (
        n_result_train == n_train_records and n_result_valid == n_valid_records
    )


# Small test on create_tf_example
@pytest.mark.create_tf_example
def test_create_tf_example(tmp_path):

    data_dir = tmp_path
    os.mkdir(os.path.join(data_dir, "tf_record_data"))
    os.mkdir(os.path.join(data_dir, "tf_record_data", "images"))

    data = {
        "video_id": 0,
        "sequence": 40258,
        "video_frame": 16,
        "sequence_frame": 16,
        "image_id": "0-16",
        "annotations": [{"x": 559, "y": 213, "width": 50, "height": 32}],
        "annotations_coco": [{"x": 559, "y": 213, "width": 50, "height": 32}],
        "annotations_pascal": [
            {"x_left": 559, "y_top": 213, "x_right": 609, "y_bottom": 245}
        ],
        "class_ids": [1],
        "class_names": ["cots"],
    }

    img = Image.new("RGB", size=(1280, 720), color=(256, 0, 0))
    img.save(os.path.join(data_dir, "tf_record_data", "images", "0-16.jpg"))
    with tf.io.gfile.GFile(
        os.path.join(data_dir, "tf_record_data", "images", "0-16.jpg"), "rb"
    ) as f:
        encoded_img = f.read()

    encoded_img_bytes_io = io.BytesIO(encoded_img)
    original_image = Image.open(encoded_img_bytes_io)

    row = pd.Series(data)
    example_tf = create_tf_example(row, data_path=data_dir)

    result = None
    for key, value in example_tf.features.feature.items():
        if key == "image/encoded":
            kind = value.WhichOneof("kind")
            result = np.array(getattr(value, kind).value)
    encoded_img_bytes_io = io.BytesIO(result)
    result_image = Image.open(encoded_img_bytes_io)

    assert original_image == result_image


@pytest.mark.create_tfrecord
def test_create_tfrecord(tmp_path):
    data_dir = tmp_path
    os.mkdir(os.path.join(data_dir, "tf_record_data"))
    os.mkdir(os.path.join(data_dir, "tf_record_data", "training_data"))
    os.mkdir(os.path.join(data_dir, "tf_record_data", "images"))
    output_path_train = os.path.join(
        data_dir, "tf_record_data", "training_data", "train.tfrecord"
    )

    data = [
        {
            "video_id": 0,
            "sequence": 40258,
            "video_frame": 16,
            "sequence_frame": 16,
            "image_id": "0-16",
            "annotations": [{"x": 559, "y": 213, "width": 50, "height": 32}],
            "annotations_coco": [
                {"x": 559, "y": 213, "width": 50, "height": 32}
            ],
            "annotations_pascal": [
                {"x_left": 559, "y_top": 213, "x_right": 609, "y_bottom": 245}
            ],
            "class_ids": [1],
            "class_names": ["cots"],
        },
        {
            "video_id": 0,
            "sequence": 40258,
            "video_frame": 16,
            "sequence_frame": 16,
            "image_id": "0-17",
            "annotations": [{"x": 559, "y": 213, "width": 50, "height": 32}],
            "annotations_coco": [
                {"x": 559, "y": 213, "width": 50, "height": 32}
            ],
            "annotations_pascal": [
                {"x_left": 559, "y_top": 213, "x_right": 609, "y_bottom": 245}
            ],
            "class_ids": [1],
            "class_names": ["cots"],
        },
    ]
    df = pd.DataFrame(data)

    img = Image.new("RGB", size=(1280, 720), color=(256, 0, 0))
    img.save(os.path.join(data_dir, "tf_record_data", "images", "0-16.jpg"))
    img2 = Image.new("RGB", size=(1280, 720), color=(256, 0, 0))
    img2.save(os.path.join(data_dir, "tf_record_data", "images", "0-17.jpg"))

    create_tfrecod(
        df=df,
        output_path=output_path_train,
        data_dir=data_dir,
        play=("None", False),
    )
    assert os.path.exists(output_path_train)
