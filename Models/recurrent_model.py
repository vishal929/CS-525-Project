# this model is a recurrent model based on LMUs for processing 12s window segments for binary classification
# since we do not require intermediate outputs, we can parallelize the lmu!

import nengo
import nengo_dl
# from nengo.utils.filter_design import cont2discrete
from scipy.signal import cont2discrete
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K


# LMU module definition in the Keras API
@tf.keras.utils.register_keras_serializable()
class KerasLMU(tf.keras.layers.Layer):
    def __init__(self, order, theta, hidden_dim, trainable_A=False, trainable_B=False, use_em=False, use_eh=False,
                 timesteps=23):
        super().__init__()
        self.order = order
        self.theta = theta
        self.hidden_dim = hidden_dim
        self.trainable_A = trainable_A
        self.trainable_B = trainable_B
        self.timesteps = 23
        # LMU's have a state (using previous memory and hidden state)
        # self.memory = tf.Variable(initial_value=tf.zeros((order,1)))
        # self.hidden_state = tf.Variable(initial_value=tf.zeros((hidden_dim,1)))
        # creating A and B matrices
        A, B = get_state_space_matrices(order, theta)
        A = tf.convert_to_tensor(A, dtype=tf.float32)
        B = tf.convert_to_tensor(B, dtype=tf.float32)
        self.A = tf.Variable(initial_value=A, trainable=trainable_A, name='A_matrix')
        self.B = tf.Variable(initial_value=B, trainable=trainable_B, name='B_vector')
        self.use_em = use_em
        self.use_eh = use_eh
        # weights for computing input state (e_x will be built at build time when we know input dim)
        if self.use_em:
            # weight for squashing prior memory
            self.e_m = self.add_weight(shape=(order, 1), initializer='glorot_uniform', trainable=True, name='mem_emb')
        if self.use_eh:
            # weight for squashing prior hidden state
            self.e_h = self.add_weight(shape=(hidden_dim, 1), initializer='glorot_uniform', trainable=True,
                                       name='hidden_emb')
        # weight for computing portions of hidden states
        # W_h, W_m (W_x will be built at build time, when we know the input dim)
        self.W_h = self.add_weight(shape=(hidden_dim, hidden_dim), initializer='glorot_uniform',
                                   trainable=True, name='hidden_weights')
        self.W_m = self.add_weight(shape=(hidden_dim, order), initializer='glorot_uniform',
                                   trainable=True, name='memory_weights')

    def build(self, input_shape):
        # need to create W_x and e_x here
        self.W_x = self.add_weight(shape=(input_shape[-1], self.hidden_dim), initializer='glorot_uniform',
                                   trainable=True, name='input_weights')
        self.e_x = self.add_weight(shape=(input_shape[-1], 1), initializer='glorot_uniform'
                                   , trainable=True, name='input_emb')
        # shape is batch x timestep x (features)

    def call(self, inputs):
        # need to compute memories for each timestep
        shape = tf.shape(inputs)
        batch = shape[0]
        memory = tf.zeros((batch, self.order, 1), dtype=tf.float32)
        # writing dummy timestep for t=-1
        hiddens = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
        hiddens = hiddens.write(0, tf.zeros((1, batch, self.hidden_dim), dtype=tf.float32))
        # iterating over the time dimension
        for i in range(1, self.timesteps + 1):
            # transpose to get (batch,hidden_dim,1)
            last_hidden = tf.transpose(hiddens.read(i - 1), perm=[1, 2, 0])
            feature = inputs[:, i - 1, :]
            # computing u
            u = tf.matmul(feature, self.e_x)
            if self.use_eh:
                u = tf.add(u, tf.matmul(last_hidden, self.e_h))
            if self.use_em:
                u = tf.add(u, tf.matmul(memory, self.e_m))
            # computing the new memory (m_t)
            # need to expand B along the batch axis for scalar multiplication
            # need to match u to B for scalar batch element wise multiplication
            # U is of shape (batch,1) so we need to reshape it to (batch,order,1)
            u = tf.expand_dims(u, axis=1)
            u = tf.repeat(u, self.order, axis=1)
            expanded_B = tf.repeat(tf.expand_dims(self.B, axis=0), batch, axis=0)
            memory = tf.add(tf.matmul(self.A, memory), tf.multiply(expanded_B, u))
            # computing the new hidden state (h_t)
            in_act = tf.matmul(feature, self.W_x)
            in_act = tf.expand_dims(in_act, axis=2)
            hidden_act = tf.matmul(self.W_h, last_hidden)
            memory_act = tf.matmul(self.W_m, memory)
            # update hidden state for this timestep
            hidden = tf.nn.leaky_relu(tf.add(in_act, tf.add(hidden_act, memory_act)))  # batch,hidden_dim,1
            # transpose to get (batch,1,hidden_dim)
            hiddens = hiddens.write(i, tf.transpose(hidden, perm=[2, 0, 1]))
        # concatenating along time dimension to get (time,batch,hidden_dim)
        hiddens = hiddens.concat()
        # removing the first timestep feature, since it was a dummy of zero values
        hiddens = hiddens[1:, :, :]
        # tranpose to get (batch,time,hidden_dim)
        hiddens = tf.transpose(hiddens, perm=[1, 0, 2])
        return hiddens

    def get_config(self):
        config = super(KerasLMU,self).get_config()
        config.update({
            "A_matrix": self.A,
            "B_vector": self.B,
            "hidden_weights": self.W_h,
            "memory_weights": self.W_m,
            "input_weights": self.W_x,
            "input_emb": self.e_x,
        })
        if self.use_em:
            # weight for squashing prior memory
            config.update({'mem_emb':self.e_m})
        if self.use_eh:
            # weight for squashing prior hidden state
            config.update({'hidden_emb':self.e_h})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# the below function is taken from the nengo state space matrices for LMU example documentation
def get_state_space_matrices(order, theta):
    Q = np.arange(order, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    B = (-1.0) ** Q[:, None] * R
    C = np.ones((1, order))
    D = np.zeros((1,))

    A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
    return A, B


# this LMU consists of 3 lmu units of hidden_dim followed by a dense output layer with sigmoid binary output
def build_lmu(order, theta, hidden_dim, num_lmus=1):
    x = input = tf.keras.Input(shape=(23, 2508))
    # reshaping input to flatten HxW dimensions
    x = tf.keras.layers.Reshape(target_shape=(23, 2508))(input)
    for i in range(num_lmus):
        x = KerasLMU(order, theta, hidden_dim)(x)

    # obtaining only the last hidden layer output
    x = tf.keras.layers.Lambda(lambda x: x[:, -1])(x)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(input, output)
    # Finally, we compute the cross-entropy loss between true labels and predicted labels to account for
    # the class imbalance between seizure and non-seizure depicting data
    # loss_func = keras.losses.categorical_crossentropy
    loss_func = tf.keras.losses.BinaryCrossentropy()
    optim = tf.keras.optimizers.RMSprop(learning_rate=0.00001)
    metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]
    model.compile(loss=loss_func, optimizer=optim, metrics=metrics)
    print('LMU model successfully built')
    return model

def remove_lambda_layer_and_sigmoid(model):
    layers = model.layers
    input_layer = layers[0]
    x = input_layer.output
    for l in layers[1:]:
        if isinstance(l, tf.keras.layers.Dense):
            if l.activation == tf.keras.activations.get('sigmoid'):
                l.activation = None
        if not isinstance(l, tf.keras.layers.Lambda):
            x = l(x)
    new_model = tf.keras.Model(input_layer.input, x)
    # print(new_model.summary())
    return new_model


# function that converts our non-spiking LMU to a spiking one in Nengo
def convert_recurrent_snn(saved_weights_directory=None, synapse=None, scale_firing_rates=1, do_training=False):
    # loading weights if they exist
    if saved_weights_directory:
        model = tf.keras.models.load_model(saved_weights_directory,custom_objects={'KerasLMU':KerasLMU})
    else:
        model = build_lmu(order=256, theta=784, hidden_dim=256, num_lmus=2)
    # need to remove dropout layers because they are not supported in nengo
    # stripped_model = remove_dropout_layers(model)
    stripped_model = remove_lambda_layer_and_sigmoid(model)
    swap_activations = {tf.nn.leaky_relu: nengo_dl.SpikingLeakyReLU()}
    if do_training:
        converted = nengo_dl.Converter(stripped_model, inference_only=False, allow_fallback=False,
                                       swap_activations=swap_activations, temporal_model=True)
    else:
        converted = nengo_dl.Converter(stripped_model, inference_only=True, allow_fallback=False,
                                       swap_activations=swap_activations, temporal_model=True)
    return converted

@nengo_dl.Converter.register(KerasLMU)
class ConvertKerasLMU(nengo_dl.converter.LayerConverter):
    def convert(self, node_id):
        A, B, W_h, W_m, W_x, e_x = tf.keras.backend.batch_get_value([self.layer.A, self.layer.B, self.layer.W_h,
                                                                     self.layer.W_m, self.layer.W_x, self.layer.e_x])
        x = nengo.Node(size_in=W_x.shape[-2])
        memory = nengo.Node(size_in=self.layer.order)
        hidden = nengo.Node(size_in=self.layer.hidden_dim)
        u = nengo.Node(size_in=1)

        # create a Nengo object representing the output of this layer node
        output = self.add_nengo_obj(node_id=node_id, activation=tf.nn.leaky_relu)

        # connect up the input of the layer node to the input in the lmu
        self.add_connection(node_id, x)

        # computing input scaling
        u_x = nengo.Connection(x, u, transform=e_x.reshape(1, np.prod(e_x.shape)), synapse=None)
        if self.layer.use_em:
            e_m = tf.keras.backend.get_value(self.layer.e_m)
            # synapse=0 to get the previous value of memory
            u_m = nengo.Connection(memory, u, transform=e_m.reshape(1, np.prod(e_m.shape)), synapse=0)
        if self.layer.use_eh:
            e_h = tf.keras.backend.get_value(self.layer.e_h)
            # synapse=0 to get previous value of hidden state
            u_h = nengo.Connection(hidden, u, transform=e_h.reshape(1, np.prod(e_h.shape)), synapse=0)

        # compute memory ops
        # synapse=0 to get previous value of memory
        m_A = nengo.Connection(memory, memory, transform=A, synapse=0)
        m_B = nengo.Connection(u, memory, transform=B, synapse=None)

        # computing hidden state
        h_x = nengo.Connection(x, hidden, transform=np.transpose(W_x), synapse=None)
        # synapse=0 to get previous hidden state
        h_h = nengo.Connection(hidden, hidden, transform=W_h, synapse=0)
        h_m = nengo.Connection(memory, hidden, transform=W_m, synapse=None)

        # connect up hidden state to output
        conn = nengo.Connection(hidden, output, transform=None, synapse=None)

        self.set_trainable(conn, False)

        return output


@nengo_dl.Converter.register(tf.keras.layers.Lambda)
class ConvertLambda(nengo_dl.converter.LayerConverter):
    def convert(self, node_id):
        # do nothing
        output = self.add_nengo_obj(node_id=node_id)
        # connect input to hidden node
        self.add_connection(node_id, output)
        return output

'''
converter = convert_recurrent_snn()
with nengo.Network() as net:
    # no need for any training
    nengo_dl.configure_settings(
        trainable=None,
        stateful=True,
        keep_history=True,
    )
    with nengo_dl.Simulator(converter.net) as sim:

        print(sim.predict(x=np.ones(shape=(2,23,2508))))

'''
'''
model = build_lmu(256,784,256,num_lmus=2)
print(model.predict_on_batch(tf.random.normal(shape=(2,23,2508),dtype=tf.float32)))


trainable_count = int(
    sum(K.count_params(layer) for layer in model.trainable_weights))
non_trainable_count = int(
    sum(K.count_params(layer) for layer in model.non_trainable_weights))

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))
'''

# converted = nengo_dl.Converter(model)

# model = build_lmu(256,784,256,num_lmus=2)
