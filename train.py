# file for training our model
from Models import model
from Data_Preparation import data_util
import tensorflow as tf

'''
Not sure if it would be better to write train and test functions from scratch, 
but Keras has built-in methods for this
'''
def train(model, tf_dataset, val_set, batch_size=32):
    # want to early stop if the validation loss does not improve for 3 consecutive epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
       patience=3
    )
    # batching and shuffling
    tf_dataset = tf_dataset.shuffle(buffer_size=2000000).batch(batch_size,num_parallel_calls=tf.data.AUTOTUNE)
    val_set = val_set.batch(batch_size,num_parallel_calls=tf.data.AUTOTUNE)
    # train_batch_size = ?, steps_per_epoch should be num_samples // train_batch_size
    # val_batch_size = ?, validation_steps should be num_val_samples // val_batch_size
    trained_model = model.fit(tf_dataset, epochs=6, verbose=2, validation_data=val_set,callbacks=[early_stop])
    return trained_model

# print gpu availability
print(tf.config.get_visible_devices())

# printing task before training
window_size = 1
batch_size=32768
leave_out = 'chb01'
print('training, batch_size = ' + str(batch_size) + ', leave_out=' + str(leave_out))

tf_dataset = data_util.tf_dataset('train',window_size=window_size,leave_out=leave_out)
val_set = data_util.tf_dataset('val')
model = model.buildModel()

trained_model = train(model,tf_dataset, val_set,batch_size = batch_size)