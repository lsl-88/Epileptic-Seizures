# Epileptic Seizures

## Overview
This repository provides the source code for the classification of interictal, preictal, and ictal during an epileptic seizure event in Electroencephalogram (EEG) dataset. This framework first uses a hyperparameter search model to perform random search of wavelet transform hyperparameters. 
Using the optimal hyperparameters obtained during the random search, the classification task was performed using a full classification model.

## Getting Started
### Installation

- Clone this repo:
```
git clone https://github.com/lsl-88/Epileptic-Seizures.git
cd Epileptic-Seizures
```

- Create the virtual environment and install the dependencies:
```
conda env create -n epileptic-seizures
conda activate epileptic-seizures
conda install --file requirements.txt
```

### Hyperparameters search
- To perform a random search of the wavelet transform hyperparameters:
```
python main.py -rs
```
The log will be save as 'hyperparameters_search_summary.csv'

### Train with specific hyperparameters
- To train and evaluate the model using specific hyperparamters of the wavelet transform, run the main script with the following settings:
  - Scales (Number of scales): -s or --scales
  - Scales Resolutions (Resolutions of the scales): -sr or --scales_res
  - Window Size (Length of window in seconds): -ws or --window_size
  - Wavelets (Type of wavelets): -w or --wavelets
    - ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']

- For instance,
```
python main.py -s 50 -sr 120 -ws 1.5 -w 'mexh'
```

## Dataset
- Download the dataset from: http://epileptologie-bonn.de/cms/upload/workgroup/lehnertz/eegdata.html
