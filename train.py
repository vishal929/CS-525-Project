# file for training our model
from Models import model,recurrent_model
from Data_Preparation import data_util
import tensorflow as tf
import os
from constants import ROOT_DIR

'''
Not sure if it would be better to write train and test functions from scratch, 
but Keras has built-in methods for this
'''


def train(model, tf_dataset, val_set, model_save_name, batch_size=32):
    # want to early stop if the validation loss does not improve for 5 consecutive epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=5
    )
    # we want to save the weights of the model while training
    checkpoint_path = os.path.join(ROOT_DIR, 'Trained Models', model_save_name + '.ckpt')

    # we only save the model with the best validation loss seen so far
    save_weights = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_weights_only=True,
                                                      verbose=1,
                                                      save_best_only=True,
                                                      monitor='val_loss')
    # batching and shuffling
    tf_dataset = tf_dataset.shuffle(buffer_size=200000).batch(batch_size, num_parallel_calls=4)
    val_set = val_set.batch(batch_size, num_parallel_calls=4)
    # train_batch_size = ?, steps_per_epoch should be num_samples // train_batch_size
    # val_batch_size = ?, validation_steps should be num_val_samples // val_batch_size
    trained_model = model.fit(tf_dataset, epochs=50, verbose=2, validation_data=val_set, callbacks=[early_stop,
                                                                                                    save_weights])
    return trained_model


# print gpu availability
print(tf.config.get_visible_devices())

# printing task before training
window_size = 12
batch_size = 32
leave_out = 'chb01'
print('training, batch_size = ' + str(batch_size) + ', leave_out=' + str(leave_out) + ', win_size: ' + str(window_size))
model_saved_name = str(leave_out) + '----' + str(window_size)

tf_dataset = data_util.tf_dataset('train', window_size=window_size, leave_out=leave_out)
val_set = data_util.tf_dataset('val',window_size=window_size,leave_out=leave_out)

if window_size==1:
    model = model.buildModel()
else:
    model = recurrent_model.build_lmu(256,784,256,num_lmus=2)

trained_model = train(model, tf_dataset, val_set, model_saved_name, batch_size=batch_size)
