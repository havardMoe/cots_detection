import tensorflow as tf
import os
import shutil
from object_detection.utils import dataset_util
import json
from PIL import Image
import io
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset(data_dir, dataset_name):
    """
    Download the dataset if not already exists
    Note -> Need to add your kaggle acc and api in kaggle.json
    file on computer for access to API
    """
    competition_dir = os.path.join(data_dir, dataset_name)
    if not os.path.exists(os.path.join(data_dir, dataset_name)):
        api = KaggleApi()
        api.authenticate()
        print("Dataset is around 15 GB may take a long time...")
        api.competition_download_files(dataset_name, path=data_dir)

        os.mkdir(competition_dir)
        with zipfile.ZipFile(
            os.path.join(data_dir, f"{dataset_name}.zip")
        ) as zip_file:
            zip_file.extractall(competition_dir)
        os.remove(os.path.join(data_dir, f"{dataset_name}.zip"))
        print("Download complete and fils unziped!")
    else:
        print("Dataset folder already present!")
    return competition_dir


def create_workspace(data_dir):
    """
    Takes in the directory to create workspace in.
    """
    if not os.path.exists(os.path.join(data_dir, "tf_record_data")):
        os.mkdir(os.path.join(data_dir, "tf_record_data"))

        train_dir = os.path.join(data_dir, "tf_record_data")
        create_dirs = ["annotations", "images", "training_data"]
        for dir in create_dirs:
            exists = os.path.exists(os.path.join(train_dir, dir))
            if not exists:
                os.mkdir(os.path.join(train_dir, dir))
        print("Workspace created.")
    else:
        print("Workspace already exists!")


def move_images(data_dir, dataset_name):
    """
    Moves the images from the dataset to the new directory
    """
    image_dir = os.path.join(data_dir, dataset_name, "train_images")
    videos = [
        v
        for v in os.listdir(
            os.path.join(data_dir, dataset_name, "train_images")
        )
        if v != ".DS_Store"
    ]
    image_paths = [
        os.path.join(image_dir, video, image)
        for video in videos
        for image in os.listdir(os.path.join(image_dir, video))
    ]

    dest = os.path.join(data_dir, "tf_record_data", "images", "train")
    for image_path in image_paths:
        sourc = image_path

        video_frame = os.path.basename(image_path)
        video_id = sourc.split(os.sep)[-2].split("_")[-1]
        image_id = f"{video_id}-{video_frame}"

        dest = os.path.join(data_dir, "tf_record_data", "images", image_id)
        shutil.move(str(sourc), str(dest))
    print(f"Images moved to {dest}")


# Fix annotation
def fix_anno(anno):
    # Was on text format, need to change it to list with dict
    # Also json.loads expects " not '
    return json.loads(anno.replace("'", '"'))


def to_pascal(row):
    """
    Changes the format on bbox from coco to pascal.
    """
    coco_list = row["annotations_coco"]
    pascal_list = []
    for coco in coco_list:
        x_left = coco["x"]
        y_top = coco["y"]
        x_right = coco["x"] + coco["width"]
        y_bottom = coco["y"] + coco["height"]
        pascal_dict = {
            "x_left": x_left,
            "y_top": y_top,
            "x_right": x_right,
            "y_bottom": y_bottom,
        }
        pascal_list.append(pascal_dict)
    return pascal_list


def box_id(row, class_id):
    """
    returns a list with ids for the bboxes
    """
    anno = row["annotations_pascal"]
    N = len(anno)
    if N == 0:
        return []
    id_list = [class_id for i in range(N)]
    return id_list


def box_class_names(row, class_name):
    anno = row["annotations_pascal"]
    N = len(anno)
    if N == 0:
        return []
    class_name_list = [class_name for i in range(N)]
    return class_name_list


def create_label_map(data_dir, class_id, class_name):
    """
    Create label map.
    """
    label_map = f"""item {{
    id:{class_id}
    name:"{class_name}"
    }} """

    if not os.path.isfile(
        os.path.join(
            data_dir, "tf_record_data", "training_data", "label_map.txt"
        )
    ):
        with open(
            os.path.join(
                data_dir, "tf_record_data", "training_data", "label_map.txt"
            ),
            "w",
        ) as f:
            f.write(label_map)
            print("Label map created!")
    else:
        print("Label map already exists!")


def split_train_val(train_meta_df, ratio=0.9):
    """
    Split into train and validation dataframes on ratio
    """
    train_df = train_meta_df.sample(frac=ratio)
    val_df = train_meta_df.drop(train_df.index)
    return train_df, val_df


'''
ref: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
ref: https://www.kaggle.com/code/khanhlvg/cots-detection-w-tensorflow-object-detection-api
ref: https://github.com/datitran/raccoon_dataset
''' # noqa


def create_tf_example(row, data_path):
    """
    Create  tf example
    """
    # Define names for row variables
    image_id = row["image_id"]
    annotations = row["annotations_pascal"]
    class_ids = row["class_ids"]
    class_names = [
        class_name.encode("utf-8") for class_name in row["class_names"]
    ]

    # Specify path to image file
    file_format = "jpg"
    file_path = os.path.join(
        data_path, "tf_record_data", "images", image_id + "." + file_format
    )

    # Get image file
    with tf.io.gfile.GFile(file_path, "rb") as f:
        encoded_img = f.read()

    encoded_img_bytes_io = io.BytesIO(encoded_img)
    image = Image.open(encoded_img_bytes_io)

    # Ready bbox for tf_example
    x_left = []
    y_top = []
    x_right = []
    y_bottom = []
    # bbox values need to be normalized
    # (not quite sure why, but everybody does it)
    for anno in annotations:
        x_left.append(anno["x_left"] / image.size[0])  # Divide by width
        y_top.append(anno["y_top"] / image.size[1])  # Divide by heigth
        x_right.append(anno["x_right"] / image.size[0])  # Divide by width
        y_bottom.append(anno["y_bottom"] / image.size[1])  # Divide by height

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(image.size[1]),
                "image/width": dataset_util.int64_feature(image.size[0]),
                "image/filename": dataset_util.bytes_feature(
                    image_id.encode("utf-8")
                ),
                "image/source_id": dataset_util.bytes_feature(
                    image_id.encode("utf-8")
                ),
                "image/encoded": dataset_util.bytes_feature(encoded_img),
                "image/format": dataset_util.bytes_feature(
                    file_format.encode("utf-8")
                ),
                "image/object/bbox/xmin": dataset_util.float_list_feature(
                    x_left
                ),
                "image/object/bbox/ymin": dataset_util.float_list_feature(
                    y_top
                ),
                "image/object/bbox/xmax": dataset_util.float_list_feature(
                    x_right
                ),
                "image/object/bbox/ymax": dataset_util.float_list_feature(
                    y_bottom
                ),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    class_names
                ),
                "image/object/class/label": dataset_util.int64_list_feature(
                    class_ids
                ),
            }
        )
    )
    return tf_example


def create_tfrecod(df, output_path, data_dir, play=("None", False)):
    """
    Create tf record
    """
    if play[1]:
        if play[0] == "valid":
            df = df.head(4)
        elif play[1] == "train":
            df = df.head(40)

    writer = tf.io.TFRecordWriter(output_path)
    for _, row in df.iterrows():
        tf_example = create_tf_example(row=row, data_path=data_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()
