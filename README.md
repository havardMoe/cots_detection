<a name="toppp"></a>
# COTS Detection

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Project in DAT550 at the University of Stavanger.  
Detecting COTS (crown-of-thorns starfish) with the use of Tensorflows Object Detection API.  
The results are available at [COTS-detectors W&B](https://wandb.ai/cots-detectors)

![Image from dataset](https://github.com/havardMoe/cots_detection/blob/1181c8e0b353cc62dfda2134c953d412c730573c/images/vis.png)

#### Table of Contents:  
- [Tech/Framework Used](#tech)  
- [Short Description](#desc)  
- [User Guide](#usr-guide)  

<a name="tech"></a>
## Tech/Framework Used
- [TensorFlow](https://www.tensorflow.org/api_docs/python/tf)
- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [Weights and Biases](https://wandb.ai/site)
- [Google Collab](https://colab.research.google.com/?utm_source=scs-index)


Script  | Description
------------- | ------------- 
[prep_data_functions.py](https://github.com/havardMoe/cots_detection/blob/59836d38673fdc7e8f771f0a619c543ef1c8ba19/code/prep_data_functions.py)  | Contains functions used to transform the data into tfrecords format.
[prep_data.ipynb](https://github.com/havardMoe/cots_detection/blob/59836d38673fdc7e8f771f0a619c543ef1c8ba19/code/prep_data.ipynb)  | Notebook used to transform the data into tfrecord format.
[train_eval_sweep.ipynb](https://github.com/havardMoe/cots_detection/blob/75ef40adb1d68bfe66d7f5fe7ef61ceca3914454/code/Colab%20Notebooks/train_eval_sweep.ipynb)  | Notebook used for training of models.
<a name="usr-guide"></a>
## User Guide:
### 1. Create conda environment:
```bash
conda env create --name cots --file=environment.yaml
```
### 2. Create a Kaggle Account and Set Up Kaggle API. 
- Create account : [Kaggle](https://www.kaggle.com/)
- Set up Kaggle API: [API guide](https://www.kaggle.com/docs/api)
### 3. Transforming Data to TFRecord Format.
- To transform the data to tfrecord format with train, validation and test files run the [prep_data.ipynb](https://github.com/havardMoe/cots_detection/blob/59836d38673fdc7e8f771f0a619c543ef1c8ba19/code/prep_data.ipynb) notebook.
### 4. Train, Evaluate and Compare Results
- Open the [train_eval_sweep.ipynb](https://github.com/havardMoe/cots_detection/blob/75ef40adb1d68bfe66d7f5fe7ef61ceca3914454/code/Colab%20Notebooks/train_eval_sweep.ipynb) notebook, and follow the given steps.
- **Note:** A detailed guide on how to download and install the Tensorflow Object Detection Api on Colab can be found [here](https://medium.com/@sourabhsinglasoftware/tensorflow-object-detection-api-setup-on-colab-e934be20f521)

# Testing and GitHub actions

Using `pre-commit` hooks, `flake8`, `black` and `pytest` are locally run on every commit. For more details on how to use `pre-commit` hooks see [here](https://github.com/iai-group/guidelines/tree/main/python#install-pre-commit-hooks).

Similarly, Github actions are used to run `flake8`, `black` and `pytest` on every push and pull request. The `pytest` results are sent to [CodeCov](https://about.codecov.io/) using their API for to get test coverage analysis. Details on Github actions are [here](https://github.com/iai-group/guidelines/blob/main/github/Actions.md).

