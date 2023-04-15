import tensorflow as tf
from tensorflow import keras
import nengo_dl
import nengo
import numpy as np
from Data_Preparation import data_util
from Models import model
from keras_data_format_converter import convert_channels_first_to_last

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

new_model = buildModel()
old_model = keras.models.load_model('Trained Models/chb01----1----seizure_number:1')
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


# converted = model.convert_snn()

# test_set = data_util.tf_dataset('test',window_size=1,leave_out='chb01')

# with converted.net:
#     # no need for any training
#     nengo_dl.configure_settings(
#         trainable=None,
#         stateful=True,
#         keep_history=True,
#     )
# with nengo_dl.Simulator(converted.net) as sim:
#     sim.compile()
#     print(sim.evaluate(x=test_set))