import tensorflow as tf
import keras
import numpy as np
import sklearn
from sklearn.utils import class_weight
from keras.layers import BatchNormalization,Conv3D, Dense, Dropout, Flatten


#Building the convolutional neural network:
def buildModel():
   
    input = keras.Input(shape = (1, 22, 114))  #consists of 22 EEG channels
    
    #We apply the ReLU activation function to each of the three 2D convolutional layers 

    #The first layer has 16 kernals of size (22, 5)
    x = keras.layers.Conv2D(16, (22, 5), strides=(1, 2), padding='valid', data_format= "channels_first", activation=tf.nn.relu)(input)
    #For each channel, MaxPool2D() takes the max value within a designated window of the input of size (1, 2) 
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format= "channels_first",  padding='same')(x)
    #Then, we normalize the max value just collected
    x = keras.layers.BatchNormalization()(x)

    #The second layer has 32 kernals of size (1, 3)
    x = keras.layers.Conv2D(32, (1, 3), strides=(1, 1), padding='valid', data_format= "channels_first", activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format= "channels_first", )(x)
    x = keras.layers.BatchNormalization()(x)

    #The third and last convolutional layer has 64 kernals also of size (1, 3)
    x = keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='valid', data_format= "channels_first", activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format= "channels_first", )(x)
    x = keras.layers.BatchNormalization()(x)

    #To create the 2 fully connected layers, we first flatten the extracted features
    x = keras.layers.Flatten()(x)

    #Dropout layers are added before both of the 2 fully connected layers at a rate of 0.5
    x = keras.layers.Dropout(0.5)(x)
    #First fully connected layer has an output size of 256 and applies the sigmoid activation function
    x = keras.layers.Dense(units=256, activation=tf.nn.sigmoid)(x)

    x = keras.layers.Dropout(0.5)(x)
    #Second fully connected layerhas an output size of 2 and applies the softmax activation function
    output = keras.layers.Dense(units=2, activation=tf.nn.softmax)(x)

    model = keras.Model(input, output)
    #Finally, we compute the cross-entropy loss between true labels and predicted labels to account for
    #the class imbalance between seizure and non-seizure depicting data
    loss_func = keras.losses.binary_crossentropy
    optim = keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(loss=loss_func, optimizer=optim, metrics=[keras.metrics.AUC()], )
    print('CNN model successfully built')
    return model

model = buildModel()


# weights = class_weight.compute_class_weight('balanced',
#                                             np.unique(y_train),
#                                             y_train)
# print(weights)

# model.fit(x_train, y_train, epochs=?, batch_size=?, class_weight=weights)