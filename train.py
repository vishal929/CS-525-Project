# file for training our model
from Models import model,recurrent_model
from Models.CustomMetrics import CombineTrainValAUC
from Data_Preparation import data_util
import tensorflow as tf
import os
from constants import ROOT_DIR
import numpy as np

'''
Not sure if it would be better to write train and test functions from scratch, 
but Keras has built-in methods for this
'''


def train(model, tf_dataset, val_set, model_save_name, batch_size=32):
    # callback for creating the monitor that is the average of train and val AUC
    create_monitor = CombineTrainValAUC()
    # want to early stop if the combined train and val AUC average does not improve for 15 consecutive epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='train_val_auc',
        patience=15
    )
    # we want to save the state of the model while training (we save the entire model to allow for continuing training)
    checkpoint_path = os.path.join(ROOT_DIR, 'Trained Models', model_save_name)

    # we only save the model with the best combined train and validation auc so far
    save_weights = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      verbose=1,
                                                      save_best_only=True,
                                                      monitor='train_val_auc')
    # batching and shuffling
    tf_dataset = tf_dataset.shuffle(buffer_size=100000).batch(batch_size, num_parallel_calls=4)
    val_set = val_set.batch(batch_size, num_parallel_calls=4)
    # train_batch_size = ?, steps_per_epoch should be num_samples // train_batch_size
    # val_batch_size = ?, validation_steps should be num_val_samples // val_batch_size
    trained_model = model.fit(tf_dataset, epochs=300, verbose=2, validation_data=val_set, callbacks=[create_monitor,
                                                                                                    early_stop,
                                                                                                    save_weights])
    return trained_model


# print gpu availability
print(tf.config.get_visible_devices())

# flag if we are training using leave-out-one patient or if we are training on a specific patient with leave_out
# if false, we are training on all data and leaving this patient's entire data as leave out
specific_patient = True

# printing task before training
window_size = 1
if window_size==1:
    batch_size = 32768
else:
    batch_size = 8192

# leave_out/specific patient
# (if specific patient flag is set, we will train,val,test for this patient)
# (if flag is not set, we will leave out this patient for testing and validation and strictly test on it)
leave_out = 'chb01'

if specific_patient:
    print('training, batch_size = ' + str(batch_size) + ', specific_patient=' + str(leave_out) + ', win_size: ' + str(
        window_size))
else:
    print('training, batch_size = ' + str(batch_size) + ', leave_out=' + str(leave_out) + ', win_size: ' + str(window_size))

model_saved_name = str(leave_out) + '----' + str(window_size)

if specific_patient:
    # getting the seizure indices for this patient
    seizure_indices = np.arange(1,data_util.get_num_seizures(leave_out)+1)
    # doing one-leave-out cross validation loop
    for seizure_index in seizure_indices:
        print('training for patient: ' + str(leave_out) + ' one seizure leave out: ' + str(seizure_index))
        model_specific_name = model_saved_name + '----seizure_number:' + str(seizure_index)
        tf_dataset,val_set,test_set = data_util.get_seizure_leave_out_data(seizure_number=seizure_index,
                                                                           window_size=window_size,
                                                                           patient=leave_out)

        if window_size == 1:
            model_to_train = model.buildModel()
        else:
            model_to_train = recurrent_model.build_lmu(256, 784, 256, num_lmus=2)
        trained_model_results = train(model_to_train, tf_dataset, val_set, model_specific_name, batch_size=batch_size)
        # evaluate on the leave out seizure
        test_set = test_set.batch(batch_size=batch_size,num_parallel_calls=4)
        results = model_to_train.evaluate(test_set)
        readable = dict(zip(model_to_train.metrics_names,results))
        print('patient: ' + str(leave_out) + ' leave_out seizure: ' + str(seizure_index) +
             ' window_size: ' +str(window_size) + ' results: ' + str(readable))
else:
    tf_dataset = data_util.tf_dataset('train', window_size=window_size, leave_out=leave_out)
    val_set = data_util.tf_dataset('val',window_size=window_size,leave_out=leave_out)

    # check if we are continuing training
    # if we are continuing, load the model, otherwise create a new one
    possible_checkpoint =os.path.join(ROOT_DIR, 'Trained Models', model_saved_name)
    if os.path.exists(possible_checkpoint):
        model_to_train = tf.keras.models.load_model(possible_checkpoint)
    else:
        if window_size==1:
            model_to_train = model.buildModel()
        else:
            model_to_train = recurrent_model.build_lmu(256,784,256,num_lmus=2)

    trained_model_hist = train(model_to_train, tf_dataset, val_set, model_saved_name, batch_size=batch_size)

    # obtaining test results by predicting on the test set
    test_set = data_util.tf_dataset('test',window_size=window_size,leave_out=leave_out)
    results = model_to_train.evaluate(test_set)
    readable = dict(zip(model_to_train.metrics_names,results))
    print('leave_out: ' + str(leave_out) + ' window_size: ' + str(window_size) + ' results: ' + str(readable))
