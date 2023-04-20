import tensorflow as tf
from tensorflow import keras
import nengo_dl
import nengo
import numpy as np
from Data_Preparation import data_util
from Models import model
from constants import ROOT_DIR
import os
from pathlib import Path
import re
#from keras_data_format_converter import convert_channels_first_to_last

#modified buildModel for channel last
def buildModel():
    input = keras.Input(shape=(22, 114, 1))  # consists of 22 EEG channels

    # We apply the ReLU activation function to each of the three 2D convolutional layers
    # I just swapped relu with leaky relu in order to avoid a "dead" gradient problem

    # The first layer has 16 kernals of size (22, 5)
    x = keras.layers.Conv2D(16, (22, 5), strides=(1, 2), padding='valid', data_format="channels_last",
                            activation=tf.nn.leaky_relu)(input)
    # For each channel, MaxPool2D() takes the max value within a designated window of the input of size (1, 2)
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format="channels_last", padding='valid')(x)
    # Then, we normalize the max value just collected
    x = keras.layers.BatchNormalization()(x)

    # The second layer has 32 kernals of size (1, 3)
    x = keras.layers.Conv2D(32, (1, 3), strides=(1, 1), padding='valid', data_format="channels_last",
                            activation=tf.nn.leaky_relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format="channels_last", )(x)
    x = keras.layers.BatchNormalization()(x)

    # The third and last convolutional layer has 64 kernals also of size (1, 3)
    x = keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='valid', data_format="channels_last",
                            activation=tf.nn.leaky_relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format="channels_last", )(x)
    x = keras.layers.BatchNormalization()(x)

    # To create the 2 fully connected layers, we first flatten the extracted features
    x = keras.layers.Flatten()(x)

    # Dropout layers are added before both of the 2 fully connected layers at a rate of 0.5
    x = keras.layers.Dropout(0.5)(x)
    # First fully connected layer has an output size of 256 and applies the sigmoid activation function
    # Changed activation to relu just to test things out
    #x = keras.layers.Dense(units=256, activation=tf.nn.sigmoid)(x)
    x = keras.layers.Dense(units=256, activation=tf.nn.leaky_relu)(x)

    x = keras.layers.Dropout(0.5)(x)
    # Second fully connected layerhas an output size of 2 and applies the softmax activation function
    #output = keras.layers.Dense(units=2, activation=tf.nn.softmax)(x)
    output = keras.layers.Dense(units=1, activation=tf.nn.sigmoid)(x)

    model = keras.Model(input, output)
    # Finally, we compute the cross-entropy loss between true labels and predicted labels to account for
    # the class imbalance between seizure and non-seizure depicting data
    #loss_func = keras.losses.categorical_crossentropy
    loss_func = keras.losses.BinaryCrossentropy()
    optim = keras.optimizers.RMSprop(learning_rate=0.00001)
    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]
    model.compile(loss=loss_func, optimizer=optim, metrics=metrics, )
    print('CNN model successfully built')
    return model

#new_model = buildModel()
#old_model = keras.models.load_model('Trained Models/chb01----1----seizure_number:1')
#old_model = convert_channels_first_to_last(old_model)
# print(old_model.summary())
# weights = [layer.get_weights() for layer in old_model.layers]
# for layer, weight in zip(old_model.layers, weights):
#     print(layer)
#     for i in weight:
#         print(i.shape)
# print(old_model.get_config())
# for layer, weight in zip(new_model.layers, weights):
#     for i in weight: print(i.shape)
#     print(layer)
#     for i in layer.get_weights(): print(i.shape)
#     layer.set_weights(weight)
# print(new_model.summary())

# globbing for saved model paths of this specific window_size
window_size=1

use_train = False
use_val = True

model_paths = Path.glob(Path(ROOT_DIR,'Trained Models'),'chb*----'+str(window_size)+'----seizure_number_*')
model_paths = list(model_paths)

# this dictionary holds mappings for each experiment i.e {patient_id:[exp_1 accuracy, exp2,accuracy]...}
patient_accs = {}

for model_path in model_paths:
    model_path = str(model_path)
    basename = os.path.basename(model_path)
    # lets get the patient id
    patient = basename[:5]
    # lets get the specific seizure to test on
    seizure_number = int(re.sub("[^0-9]","",basename[-2:]))
    # specify timesteps to repeat input for snn
    timesteps = 50

    print('testing snn for patient: ' + str(patient) + ' on seizure number: ' + str(seizure_number))

    converted = model.convert_snn(model_path)

    # we can throw out train and val, no need for that in evaluation
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


    # repeating examples and labels for a certain number of timesteps
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

        validation_examples = validation_examples.reshape(num_val_examples, 1, -1)
        validation_examples = np.tile(validation_examples, (1, timesteps, 1))
        validation_labels = np.expand_dims(validation_labels, axis=-1).repeat(num_features, axis=-1)
        validation_labels = np.expand_dims(validation_labels, axis=1).repeat(timesteps, axis=1)

    with converted.net:
        # no need for any training
        nengo_dl.configure_settings(
            trainable=None,
            stateful=False,
            keep_history=True,
        )
        with nengo_dl.Simulator(converted.net,progress_bar=True,minibatch_size=num_examples) as sim:
            #print(converted.net.probeable)
            '''
            sim.compile(loss={converted.outputs[converted.model.output]:keras.losses.BinaryCrossentropy(from_logits=True)},
                        metrics={
                            converted.outputs[converted.model.output]:keras.metrics.TruePositives(name='tp'),
                            converted.outputs[converted.model.output]:keras.metrics.FalsePositives(name='fp'),
                            converted.outputs[converted.model.output]:keras.metrics.TrueNegatives(name='tn'),
                            converted.outputs[converted.model.output]:keras.metrics.FalseNegatives(name='fn'),
                            converted.outputs[converted.model.output]:keras.metrics.BinaryAccuracy(name='accuracy'),
                            converted.outputs[converted.model.output]:keras.metrics.Precision(name='precision'),
                            converted.outputs[converted.model.output]:keras.metrics.Recall(name='recall'),
                            converted.outputs[converted.model.output]:keras.metrics.AUC(name='auc'),
                        })
            '''
            test_pred = sim.predict(x=examples,n_steps=timesteps)

    if use_train:
        with converted.net:
            # no need for any training
            nengo_dl.configure_settings(
                trainable=None,
                stateful=False,
                keep_history=True,
            )
            with nengo_dl.Simulator(converted.net, progress_bar=True, minibatch_size=num_train_examples) as sim:
                train_pred = sim.predict(x=train_examples, n_steps=timesteps)

    if use_val:
        with converted.net:
            # no need for any training
            nengo_dl.configure_settings(
                trainable=None,
                stateful=False,
                keep_history=True,
            )
            with nengo_dl.Simulator(converted.net, progress_bar=True, minibatch_size=num_val_examples) as sim:
                val_pred = sim.predict(x=validation_examples, n_steps=timesteps)

    test_pred = list(test_pred.values())[0]

    test_pred = np.squeeze(test_pred)
    # decoding by taking the mean activation
    test_pred = np.mean(test_pred,axis=-1)

    # if decoded value < 0 , then we give 0, if decoded value > 0, we give 1
    test_pred = np.greater_equal(test_pred,0).astype(np.int32)
    labels = np.squeeze(labels[:,0,0])

    if use_train:
        train_pred = list(train_pred.values())[0]

        train_pred = np.squeeze(train_pred)
        # decoding by taking the mean activation
        train_pred = np.mean(train_pred, axis=-1)

        # if decoded value < 0 , then we give 0, if decoded value > 0, we give 1
        train_pred = np.greater_equal(train_pred, 0).astype(np.int32)
        train_labels = np.squeeze(train_labels[:, 0, 0])

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

    if use_val:
        val_pred = list(val_pred.values())[0]

        val_pred = np.squeeze(val_pred)
        # decoding by taking the mean activation
        val_pred = np.mean(val_pred, axis=-1)

        # if decoded value < 0 , then we give 0, if decoded value > 0, we give 1
        val_pred = np.greater_equal(val_pred, 0).astype(np.int32)
        val_labels = np.squeeze(validation_labels[:, 0, 0])

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

    # test accuracy
    test_acc = np.equal(test_pred, labels).astype(np.int32).mean()
    print('test accuracy: ' + str(test_acc))

    # storing detection accuracy in dictionary so we can compute mean at the end
    if patient in patient_accs:
        (patient_accs[patient]).append(test_acc)
    else:
        patient_accs[patient] = [test_acc]

    #print(pred.shape)
    #print(pred)
    #print('pred shape: ' + str(pred.shape))
    #print('labels shape: ' + str(labels.shape))
    #print('accuracy: ' + str((np.equal(np.round(pred),labels)).astype(np.int32).mean()))



# computing patient accuracies for each experiment
for patient in patient_accs:
    total_acc = np.array(patient_accs[patient]).mean()
    print('testing accuracy on all experiments for patient: ' + str(patient) + ' acc: ' + str(total_acc))

