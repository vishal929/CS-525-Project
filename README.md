# CS-525-Project

In this project we train an epileptic seizure classifier on the CHB-MIT EEG dataset.
This is a binary classification task that decides between ictal and interictal labels.
This is also an imbalanced classification task, so we use sample weighting in the cross entropy loss function.
train.py holds the logic to train ANNs (artificial neural networks) while test_snn.py automatically generates results 
from conversion to SNN (spiking neural networks).

Models are based on the following work, with the exception that we use average pooling instead of max pooling:
http://levinkuhlmann.byethost3.com/pdfs/2018TruongCNNSeizureDetection.pdf

Our processed data is provided here:
https://rutgersconnect-my.sharepoint.com/:u:/r/personal/vrp55_scarletmail_rutgers_edu/Documents/Processed_Data.tar.gz?csf=1&web=1&e=AkHOPg

Our trained models are provided here:
https://rutgersconnect-my.sharepoint.com/:u:/r/personal/vrp55_scarletmail_rutgers_edu/Documents/Trained_Models_AvgPool.tar.gz?csf=1&web=1&e=nrGR9p


We use a python 3.8 conda environment with Tensorflow 2.11.0 and nengo-dl version 3.6.


