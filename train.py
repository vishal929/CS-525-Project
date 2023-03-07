# file for training our model
from Models import model
from Data_Preparation import data_util

'''
Not sure if it would be better to write train and test functions from scratch, 
but Keras has built-in methods for this
'''
def train(model, tf_dataset, val_set):
    # train_batch_size = ?, steps_per_epoch should be num_samples // train_batch_size
    # val_batch_size = ?, validation_steps should be num_val_samples // val_batch_size
    trained_model = model.fit(tf_dataset, epochs=6, verbose=2, validation_data=val_set, steps_per_epoch=None, 
              validation_steps=None)
    return trained_model

'''
Want something like the below (also need to run validation on the data after every epoch)
Essentially, we train for each epoch and then test on validation data
We have some "patience" so if the val score does not increase for lets say 5 consecutive epochs, then we stop training
We save only the best model based on validation so far
'''
tf_dataset = data_util.tf_dataset('train')
val_set = data_util.tf_dataset('val')
model = model.buildModel()

trained_model = train(model,tf_dataset, val_set)