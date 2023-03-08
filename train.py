# file for training our model
from Models import model
from Data_Preparation import data_util
import tensorflow as tf

'''
Not sure if it would be better to write train and test functions from scratch, 
but Keras has built-in methods for this
'''
def train(model, tf_dataset, val_set, batch_size=3):

    # batching and shuffling
    tf_dataset = tf_dataset.shuffle(buffer_size=1000000).batch(batch_size,num_parallel_calls=tf.data.AUTOTUNE)
    val_set = val_set.batch(batch_size,num_parallel_calls=tf.data.AUTOTUNE)
    # train_batch_size = ?, steps_per_epoch should be num_samples // train_batch_size
    # val_batch_size = ?, validation_steps should be num_val_samples // val_batch_size
    trained_model = model.fit(tf_dataset, epochs=6, verbose=1, validation_data=val_set)
    return trained_model

# test training on val for now
tf_dataset = data_util.tf_dataset('val')
val_set = data_util.tf_dataset('val')
model = model.buildModel()

trained_model = train(model,tf_dataset, val_set)