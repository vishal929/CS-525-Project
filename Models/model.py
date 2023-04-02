import tensorflow as tf
from tensorflow import keras
import nengo_dl
import nengo

# around 92k params
# Building the convolutional neural network:
def buildModel():
    input = keras.Input(shape=(1, 22, 114))  # consists of 22 EEG channels

    # We apply the ReLU activation function to each of the three 2D convolutional layers
    # I just swapped relu with leaky relu in order to avoid a "dead" gradient problem

    # The first layer has 16 kernals of size (22, 5)
    x = keras.layers.Conv2D(16, (22, 5), strides=(1, 2), padding='valid', data_format="channels_first",
                            activation=tf.nn.leaky_relu)(input)
    # For each channel, MaxPool2D() takes the max value within a designated window of the input of size (1, 2)
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format="channels_first", padding='valid')(x)
    # Then, we normalize the max value just collected
    x = keras.layers.BatchNormalization()(x)

    # The second layer has 32 kernals of size (1, 3)
    x = keras.layers.Conv2D(32, (1, 3), strides=(1, 1), padding='valid', data_format="channels_first",
                            activation=tf.nn.leaky_relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format="channels_first", )(x)
    x = keras.layers.BatchNormalization()(x)

    # The third and last convolutional layer has 64 kernals also of size (1, 3)
    x = keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='valid', data_format="channels_first",
                            activation=tf.nn.leaky_relu)(x)
    x = keras.layers.MaxPool2D(pool_size=(1, 2), data_format="channels_first", )(x)
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

# helper function to explicitly strip a model of its dropout layers for nengo conversion
def remove_dropout_layers(model):
    layers = model.layers
    input_layer = layers[0]
    x = input_layer.output
    for l in layers[1:]:
        if not isinstance(l,keras.layers.Dropout):
            x = l(x)
    new_model = keras.Model(input_layer.input,x)
    print(new_model.summary())
    return new_model

# loading a model from a saved checkpoint
# model path should be something like '...../Trained Models/chb01-1.ckpt'
# chb01 is the leave-out and 1 is the window_size
def load_model(model_path):
    model = buildModel()
    model.load_weights(model_path)

# function that converts this keras model into an snn in nengo
def convert_snn(saved_weights_directory=None):
    model = buildModel()
    # loading weights if they exist
    if saved_weights_directory:
        model.load_weights(saved_weights_directory)
    # need to remove dropout layers because they are not supported in nengo
    stripped_model = remove_dropout_layers(model)
    swap_activations = {tf.nn.leaky_relu:nengo_dl.SpikingLeakyReLU()}
    converted = nengo_dl.Converter(stripped_model,max_to_avg_pool=True,inference_only=True,allow_fallback=False,
                                   swap_activations=swap_activations)
    print(converted.verify())
    return converted

converted = convert_snn()

#with nengo_dl.Simulator(converted.net) as sim:
#    sim.run(3)