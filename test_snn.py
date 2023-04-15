import tensorflow as tf
from tensorflow import keras
import nengo_dl
import nengo
import numpy as np
from Data_Preparation import data_util
from Models import model
from constants import ROOT_DIR
import os
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


seizure_number=1
window_size=1
leave_out = 'chb01'
timesteps = 20

converted = model.convert_snn(os.path.join(ROOT_DIR,'Trained Models','chb01----1----seizure_number_1'))

# we can throw out train and val, no need for that in evaluation
_,_,test_set = data_util.get_seizure_leave_out_data(seizure_number=1,window_size=window_size,patient=leave_out)

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
examples = np.expand_dims(examples,axis=1).repeat(timesteps,axis=1).reshape((num_examples,timesteps,2508))
#labels = np.expand_dims(labels,axis=-1).repeat(2508,axis=-1)
labels = np.expand_dims(labels,axis=1).repeat(timesteps,axis=1)
labels = np.expand_dims(labels,axis=-1)

# examples should be (num_examples,timesteps,22x114=2508)
print(examples.shape)
print(labels.shape)

#print(converted.verify(inputs=np.ones((1,1,22,114))))

with converted.net:
    print(converted.net)
    print(converted.net.inputs)
    print(converted.net.outputs)
    # no need for any training
    nengo_dl.configure_settings(
        trainable=None,
        stateful=False,
        keep_history=False,
    )
    with nengo_dl.Simulator(converted.net,progress_bar=True) as sim:
        #print(converted.net.probeable)
        sim.compile(loss=keras.losses.BinaryCrossentropy(),
                    metrics=[
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ])
        pred = sim.predict(examples)
        #print(sim.predict(x=np.expand_dims(examples[0,:,:],axis=0)))
        #print(sim.evaluate(x=examples,y=labels))
pred = list(pred.values())[0]
print(pred.shape)
pred = np.squeeze(pred)
labels = np.squeeze(labels[:,0,:])
print('pred shape: ' + str(pred.shape))
print('labels shape: ' + str(labels.shape))
print((np.equal(np.round(pred),labels)).astype(np.int32).sum())
#print(list(pred.values())[0].shape)
