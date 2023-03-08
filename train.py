# file for training our model
from Models import model
from Data_Preparation import data_util
import tensorflow as tf

'''
Not sure if it would be better to write train and test functions from scratch, 
but Keras has built-in methods for this
'''
def train(model, tf_dataset, val_set, batch_size=3):

    # grabbing class imbalances so we can set this on the fly for training
    #num_interictal, num_ictal = data_util.get_class_counts(tf_dataset)
    #imbalance = -(num_interictal//-num_ictal)
    imbalance=2
    # we want ictal segments to be weighted more for how imbalanced they are
    weight = {0:1, 1:imbalance}

    # batching
    tf_dataset = tf_dataset.batch(batch_size)
    val_set = val_set.batch(batch_size)
    # train_batch_size = ?, steps_per_epoch should be num_samples // train_batch_size
    # val_batch_size = ?, validation_steps should be num_val_samples // val_batch_size
    trained_model = model.fit(tf_dataset, epochs=6, verbose=1, validation_data=val_set)
    return trained_model

# test training on val for now
tf_dataset = data_util.tf_dataset('val')
val_set = data_util.tf_dataset('val')
model = model.buildModel()

trained_model = train(model,tf_dataset, val_set)