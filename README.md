# Deep-Celebs: Deep Learning for Celebrity Attribute Recognition and Landmark prediction
This repository contains the code for our Deep Learning final project. It contains the following files:
* `models/` contains the models used for training and testing.
* `attributes.py` contains the code for training the attribute recognition model.
* `landmarks.py` contains the code for training the landmark prediction model.
* `utils.py` contains the code for loading the data and preprocessing it.
* `multi_task.py` contains the code for training the multi-task model.
* `plots.py` contains the code for plotting the results.
* `eval_attributes.py` contains the code for evaluating the attribute recognition model.
* `eval_landmarks.py` contains the code for evaluating the landmark prediction model.

## Before you start
Please make sure to set `download=True`in the `get_data_loaders` function in `utils.py` to download the data. 
This will take a while, so please be patient. The data will be downloaded to the `data/` folder.

Also please make sure to install all the required packages. You can do this by running `pip install -r requirements.txt`.

## Training
To train the models, please run the following commands:
* `python attributes.py` to train the attribute recognition model.
* `python landmarks.py` to train the landmark prediction model.
* `python multi_task.py` to train the multi-task model.
* `python eval_attributes.py` to evaluate the attribute recognition model.
* `python eval_landmarks.py` to evaluate the landmark prediction model.
