# file for training our model
from Models import model
from Data_Preparation import data_util

def train(model, tf_dataset):
    pass

'''
Want something like the below (also need to run validation on the data after every epoch)
Essentially, we train for each epoch and then test on validation data
We have some "patience" so if the val score does not increase for lets say 5 consecutive epochs, then we stop training
We save only the best model based on validation so far
tf_dataset = data_util.tf_dataset('train')
model = model.buildModel()

train(model,tf_dataset)
'''