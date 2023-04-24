import tensorflow as tf
from tensorflow import keras
import nengo_dl
import nengo
import numpy as np
from Data_Preparation import data_util
from Models import model,recurrent_model
from constants import ROOT_DIR
import os
from pathlib import Path
import re
#from keras_data_format_converter import convert_channels_first_to_last

# globbing for saved model paths of this specific window_size
window_size=1

use_train = False
use_val = False
do_train = False

model_paths = Path.glob(Path(ROOT_DIR,'Trained_Models'),'chb*----'+str(window_size)+'----seizure_number_*')
model_paths = list(model_paths)

# this dictionary holds mappings for each experiment i.e {patient_id:[exp_1 accuracy, exp2,accuracy]...}
patient_accs = {}
non_snn_patient_accs = {}

for model_path in model_paths:
    model_path = str(model_path)
    basename = os.path.basename(model_path)
    # lets get the patient id
    patient = basename[:5]
    # lets get the specific seizure to test on
    seizure_number = int(re.sub("[^0-9]","",basename[-2:]))
    # specify timesteps to repeat input for snn
    if window_size==1:
        timesteps = 200
    else:
        timesteps = 23
    # specify synaptic filter
    synapse=0.01
    # specify scaling of firing rates
    scale_firing_rates=250

    print('testing snn for patient: ' + str(patient) + ' on seizure number: ' + str(seizure_number))

    if window_size==12:
       converted = recurrent_model.convert_recurrent_snn(model_path,synapse,scale_firing_rates,do_train)
    else:
        converted = model.convert_snn(model_path,do_train)
    non_snn_model = keras.models.load_model(model_path)

    train_set,validation_set,test_set = data_util.get_seizure_leave_out_data(seizure_number=seizure_number,
                                                        window_size=window_size,
                                                        patient=patient)

    if not use_val: del(validation_set)
    if not use_train: del(train_set)

    # split into examples and labels
    examples = test_set.map(lambda example,label: example)
    labels = test_set.map(lambda example,label: label)


    # convert dataset to numpy lists (no problem for evaluation)
    examples = list(examples.as_numpy_iterator())
    labels = list(labels.as_numpy_iterator())

    # need to repeat inputs for some timesteps for snn
    num_examples = len(examples)
    num_features = 22*114

    examples = np.stack(examples,axis=0)
    labels = np.stack(labels)


    # repeating examples and labels for a certain number of timesteps only if our window size is not 12
    if window_size==1:
        examples = examples.reshape(num_examples,1,-1)
        examples = np.tile(examples, (1,timesteps,1))
        labels = np.expand_dims(labels,axis=-1).repeat(num_features,axis=-1)
        labels = np.expand_dims(labels,axis=1).repeat(timesteps,axis=1)

    # examples should be (num_examples,timesteps,22x114=2508)
    #print(examples.shape)
    #print(labels.shape)

    #print(converted.verify(inputs=np.ones((1,1,22,114))))

    if use_train:
        # prepping train data
        train_examples = train_set.map(lambda example, label, imbalance: example)
        train_labels = train_set.map(lambda example, label, imbalance: label)

        train_examples = list(train_examples.as_numpy_iterator())
        train_labels = list(train_labels.as_numpy_iterator())

        num_train_examples = len(train_examples)
        train_examples = np.stack(train_examples, axis=0)
        train_labels = np.stack(train_labels)

        if window_size==1:
            train_examples = train_examples.reshape(num_train_examples, 1, -1)
            train_examples = np.tile(train_examples, (1, timesteps, 1))
            train_labels = np.expand_dims(train_labels, axis=-1).repeat(num_features, axis=-1)
            train_labels = np.expand_dims(train_labels, axis=1).repeat(timesteps, axis=1)

    if use_val:
        # prepping val data
        validation_examples = validation_set.map(lambda example, label: example)
        validation_labels = validation_set.map(lambda example, label: label)

        validation_examples = list(validation_examples.as_numpy_iterator())
        validation_labels = list(validation_labels.as_numpy_iterator())

        num_val_examples = len(validation_examples)

        validation_examples = np.stack(validation_examples, axis=0)
        validation_labels = np.stack(validation_labels)

        if window_size==1:
            validation_examples = validation_examples.reshape(num_val_examples, 1, -1)
            validation_examples = np.tile(validation_examples, (1, timesteps, 1))
            validation_labels = np.expand_dims(validation_labels, axis=-1).repeat(num_features, axis=-1)
            validation_labels = np.expand_dims(validation_labels, axis=1).repeat(timesteps, axis=1)

    if use_train:
        with converted.net:
            # no need for any training
            nengo_dl.configure_settings(
                trainable=False,
                inference_only=True,
                stateful=False,
                keep_history=True,
            )
            with nengo_dl.Simulator(converted.net, progress_bar=True, minibatch_size=8) as sim:
                sim.compile(
                    loss={converted.outputs[converted.model.output]: keras.losses.BinaryCrossentropy(from_logits=True)},
                    metrics={
                        converted.outputs[converted.model.output]: keras.metrics.TruePositives(name='tp'),
                        converted.outputs[converted.model.output]: keras.metrics.FalsePositives(name='fp'),
                        converted.outputs[converted.model.output]: keras.metrics.TrueNegatives(name='tn'),
                        converted.outputs[converted.model.output]: keras.metrics.FalseNegatives(name='fn'),
                        converted.outputs[converted.model.output]: keras.metrics.BinaryAccuracy(name='accuracy'),
                        converted.outputs[converted.model.output]: keras.metrics.Precision(name='precision'),
                        converted.outputs[converted.model.output]: keras.metrics.Recall(name='recall'),
                        converted.outputs[converted.model.output]: keras.metrics.AUC(name='auc'),
                    })
                if do_train:
                    sim.fit(x=train_examples,y=train_labels,n_steps=timesteps)
                train_pred = sim.predict(x=train_examples, n_steps=timesteps)
        non_snn_train_pred = non_snn_model.predict(train_examples[:, 0, :].reshape(-1, 1, 22, 114), batch_size=32)
        non_snn_train_pred = np.round(non_snn_train_pred.flatten())
        # non_snn_train_pred = np.greater(non_snn_train_pred,0.5).astype(np.int32)
        print('num train examples: ' + str(num_train_examples))

    with converted.net:
        # no need for any training
        nengo_dl.configure_settings(
            trainable=False,
            inference_only=True,
            stateful=False,
            keep_history=True,
        )
        with nengo_dl.Simulator(converted.net,progress_bar=True,minibatch_size=num_examples) as sim:
            test_pred = sim.predict(x=examples,n_steps=timesteps)

    non_snn_test_pred = non_snn_model.predict(examples[:,0,:].reshape(-1,1,22,114),batch_size=32)
    non_snn_test_pred = np.round(non_snn_test_pred.flatten())
    print('non_snn_test_pred shape: ' + str(non_snn_test_pred.shape))
    #non_snn_test_pred = np.greater(non_snn_test_pred,0.5).astype(np.int32)

    print('num test examples: ' + str(num_examples))


    if use_val:
        with converted.net:
            # no need for any training
            nengo_dl.configure_settings(
                trainable=False,
                inference_only=True,
                stateful=False,
                keep_history=True,
            )
            with nengo_dl.Simulator(converted.net, progress_bar=True, minibatch_size=8) as sim:
                val_pred = sim.predict(x=validation_examples, n_steps=timesteps)
        non_snn_val_pred = non_snn_model.predict(validation_examples[:, 0, :].reshape(-1,1,22,114), batch_size=32)
        non_snn_val_pred = np.round(non_snn_val_pred.flatten())
        print('non_snn val_pred shape: ' + str(non_snn_val_pred.shape))
        # converting non_snn_val_pred sigmoid values to actual predictions
        #non_snn_val_pred = np.greater(non_snn_val_pred,0.5).astype(np.int32)
        print('num val examples: ' + str(num_val_examples))


    test_pred = list(test_pred.values())[0]

    test_pred = np.squeeze(test_pred)
    # decoding by taking the mean activation
    test_pred = np.mean(test_pred,axis=-1).flatten()

    # if decoded value <0 , then we give 0, if decoded value >0, we give 1
    test_pred = np.greater(test_pred,0).astype(np.int32)

    # getting num predictions (in case input was truncated due to batch size)
    num_test = test_pred.shape[0]
    labels = np.squeeze(labels[:num_test,0,0])

    # truncating keras output if needed
    non_snn_test_pred = non_snn_test_pred[:num_test]

    if use_train:
        train_pred = list(train_pred.values())[0]

        train_pred = np.squeeze(train_pred)
        # decoding by taking the mean activation
        train_pred = np.mean(train_pred, axis=-1).flatten()

        # if decoded value <= 0 , then we give 0, if decoded value >0, we give 1 (this is because we removed the sigmoid)
        train_pred = np.greater(train_pred,0).astype(np.int32)
        # getting num predictions(in case input was truncated due to batch size)
        num_train = train_pred.shape[0]
        train_labels = np.squeeze(train_labels[:num_train, 0, 0])

        # getting false positives,false negatives, true negatives, and true positives
        train_fp = np.logical_and(np.equal(train_pred, 1), np.equal(train_labels, 0)).astype(np.int32).sum()
        train_tp = np.logical_and(np.equal(train_pred, 1), np.equal(train_labels, 1)).astype(np.int32).sum()
        train_fn = np.logical_and(np.equal(train_pred, 0), np.equal(train_labels, 1)).astype(np.int32).sum()
        train_tn = np.logical_and(np.equal(train_pred, 0), np.equal(train_labels, 0)).astype(np.int32).sum()
        print('train false positive: ' + str(train_fp))
        print('train true positive: ' + str(train_tp))
        print('train false negative: ' + str(train_fn))
        print('train true negative: ' + str(train_tn))

        # computing accuracy
        train_accuracy = np.equal(train_pred, train_labels).astype(np.int32).mean()
        print('train accuracy: ' + str(train_accuracy))

        # computing precision
        train_precision = train_tp / (train_fp + train_tp)
        print('train precision: ' + str(train_precision))

        # computing recall
        train_recall = train_tp / (train_tp + train_fn)
        print('train recall: ' + str(train_recall))

        # truncating keras output if needed
        non_snn_train_pred = non_snn_train_pred[:num_train]

        # getting false positives,false negatives, true negatives, and true positives
        non_snn_train_fp = np.logical_and(np.equal(non_snn_train_pred, 1), np.equal(train_labels, 0)).astype(np.int32).sum()
        non_snn_train_tp = np.logical_and(np.equal(non_snn_train_pred, 1), np.equal(train_labels, 1)).astype(np.int32).sum()
        non_snn_train_fn = np.logical_and(np.equal(non_snn_train_pred, 0), np.equal(train_labels, 1)).astype(np.int32).sum()
        non_snn_train_tn = np.logical_and(np.equal(non_snn_train_pred, 0), np.equal(train_labels, 0)).astype(np.int32).sum()
        print('non_snn train false positive: ' + str(non_snn_train_fp))
        print('non_snn train true positive: ' + str(non_snn_train_tp))
        print('non_snn train false negative: ' + str(non_snn_train_fn))
        print('non_snn train true negative: ' + str(non_snn_train_tn))

        # computing accuracy
        non_snn_train_accuracy = np.equal(non_snn_train_pred, train_labels).astype(np.int32).mean()
        print('non_snn_train accuracy: ' + str(non_snn_train_accuracy))

        # computing precision
        non_snn_train_precision = non_snn_train_tp / (non_snn_train_fp + non_snn_train_tp)
        print('non_snn train precision: ' + str(non_snn_train_precision))

        # computing recall
        non_snn_train_recall = non_snn_train_tp / (non_snn_train_tp + non_snn_train_fn)
        print('non_snn_train recall: ' + str(non_snn_train_recall))

    if use_val:
        val_pred = list(val_pred.values())[0]

        val_pred = np.squeeze(val_pred)
        # decoding by taking the mean activation
        val_pred = np.mean(val_pred, axis=-1).flatten()

        # if decoded value <= 0 , then we give 0, if decoded value > 0, we give 1
        val_pred = np.greater(val_pred,0).astype(np.int32)
        # getting num predictions in case input was truncated due to batch size
        num_val = val_pred.shape[0]
        val_labels = np.squeeze(validation_labels[:num_val, 0, 0])

        # getting false positives,false negatives, true negatives, and true positives
        val_fp = np.logical_and(np.equal(val_pred, 1), np.equal(val_labels, 0)).astype(np.int32).sum()
        val_tp = np.logical_and(np.equal(val_pred, 1), np.equal(val_labels, 1)).astype(np.int32).sum()
        val_fn = np.logical_and(np.equal(val_pred, 0), np.equal(val_labels, 1)).astype(np.int32).sum()
        val_tn = np.logical_and(np.equal(val_pred, 0), np.equal(val_labels, 0)).astype(np.int32).sum()
        print('val false positive: ' + str(val_fp))
        print('val true positive: ' + str(val_tp))
        print('val false negative: ' + str(val_fn))
        print('val true negative: ' + str(val_tn))

        # computing accuracy
        val_accuracy = np.equal(val_pred, val_labels).astype(np.int32).mean()
        print('val accuracy: ' + str(val_accuracy))

        # computing precision
        val_precision = val_tp / (val_fp + val_tp)
        print('val precision: ' + str(val_precision))

        # computing recall
        val_recall = val_tp / (val_tp + val_fn)
        print('val recall: ' + str(val_recall))

        # truncating keras output if needed
        non_snn_val_pred = non_snn_val_pred[:num_val]

        # getting false positives,false negatives, true negatives, and true positives
        non_snn_val_fp = np.logical_and(np.equal(non_snn_val_pred, 1), np.equal(val_labels, 0)).astype(np.int32).sum()
        non_snn_val_tp = np.logical_and(np.equal(non_snn_val_pred, 1), np.equal(val_labels, 1)).astype(np.int32).sum()
        non_snn_val_fn = np.logical_and(np.equal(non_snn_val_pred, 0), np.equal(val_labels, 1)).astype(np.int32).sum()
        non_snn_val_tn = np.logical_and(np.equal(non_snn_val_pred, 0), np.equal(val_labels, 0)).astype(np.int32).sum()
        print('non_snn_val false positive: ' + str(non_snn_val_fp))
        print('non_snn_val true positive: ' + str(non_snn_val_tp))
        print('non_snn_val false negative: ' + str(non_snn_val_fn))
        print('non_snn_val true negative: ' + str(non_snn_val_tn))

        # computing accuracy
        non_snn_val_accuracy = np.equal(non_snn_val_pred, val_labels).astype(np.int32).mean()
        print('non_snn val accuracy: ' + str(non_snn_val_accuracy))

        # computing precision
        non_snn_val_precision = non_snn_val_tp / (non_snn_val_fp + non_snn_val_tp)
        print('non_snn val precision: ' + str(non_snn_val_precision))

        # computing recall
        non_snn_val_recall = non_snn_val_tp / (non_snn_val_tp + non_snn_val_fn)
        print('non_snn val recall: ' + str(non_snn_val_recall))

    # test accuracy
    test_acc = np.equal(test_pred, labels).astype(np.int32).mean()
    print('snn test accuracy: ' + str(test_acc))

    # storing detection accuracy in dictionary so we can compute mean at the end
    if patient in patient_accs:
        (patient_accs[patient]).append(test_acc)
    else:
        patient_accs[patient] = [test_acc]

    # test accuracy
    non_snn_test_acc = np.equal(non_snn_test_pred, labels).astype(np.int32).mean()
    print('non_snn test accuracy: ' + str(non_snn_test_acc))

    # storing detection accuracy in dictionary so we can compute mean at the end
    if patient in non_snn_patient_accs:
        (non_snn_patient_accs[patient]).append(non_snn_test_acc)
    else:
        non_snn_patient_accs[patient] = [non_snn_test_acc]

    #print(pred.shape)
    #print(pred)
    #print('pred shape: ' + str(pred.shape))
    #print('labels shape: ' + str(labels.shape))
    #print('accuracy: ' + str((np.equal(np.round(pred),labels)).astype(np.int32).mean()))



# computing patient accuracies for each experiment
for patient in patient_accs:
    total_acc = np.array(patient_accs[patient]).mean()
    print('snn testing accuracy on all experiments for patient: ' + str(patient) + ' acc: ' + str(total_acc))

    non_snn_total_acc = np.array(non_snn_patient_accs[patient]).mean()
    print('non_snn testing accuracy on all experiments for patient: ' + str(patient) + ' acc: ' + str(non_snn_total_acc))

