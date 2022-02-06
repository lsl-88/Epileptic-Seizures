# Epileptic-Seizures

## Overview
This repository provides the source code for the classification of epileptic seizures using ResNet. Wavelet transformation is used to transform the time series EEG signal to scalograms, and the residual block of the ResNet is then used to perform the hyperparameters search of the aforementioned signal processing technique.

## Dataset

Kaggle - https://www.kaggle.com/chaditya95/epileptic-seizures-dataset

## Hyperparameter search of wavelet transform

To perform a random hyperparameter search of the wavelet transform, do the following:

```
python main.py -rs 
```
or 
```
python main.py --random_search
```
The default iterations for search is 10. Defined as 'SEARCH_RANGE' in the main script. Once done, it is saved as 'hyperparameters_search_summary.csv' in the main directory.

## Train the model with specific wavelet transform hyperparameters

For instance, 
- scales: 30
- num_scales: 120
- window_size: 2.5
- wavelet: 'mexh'

```
python main.py -s 30 -n 120 -ws 2.5 -w 'mexh
```
or
```
python main.py --scales 30 --num_scales 120 --window_size 2.5 --wavelets 'mexh
```
