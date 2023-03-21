# this model is a recurrent model based on LMUs for processing 12s window segments for binary classification
# since we do not require intermediate outputs, we can parallelize the lmu!

#import nengo
#import nengo_dl
#from nengo.utils.filter_design import cont2discrete
from scipy.signal import cont2discrete
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

# LMU module definition in the Keras API
class KerasLMU(tf.keras.layers.Layer):
    def __init__(self,order,theta,hidden_dim,trainable_A=False,trainable_B=False,use_em=False,use_eh=False):
        super().__init__()
        self.order = order
        self.theta =theta
        self.hidden_dim = hidden_dim
        self.trainable_A = trainable_A
        self.trainable_B = trainable_B
        # LMU's have a state (using previous memory and hidden state)
        #self.memory = tf.Variable(initial_value=tf.zeros((order,1)))
        #self.hidden_state = tf.Variable(initial_value=tf.zeros((hidden_dim,1)))
        # creating A and B matrices
        A,B = get_state_space_matrices(order,theta)
        A = tf.convert_to_tensor(A,dtype=tf.float32)
        B = tf.convert_to_tensor(B,dtype=tf.float32)
        self.A = tf.Variable(initial_value=A,trainable=trainable_A,name='A_matrix')
        self.B = tf.Variable(initial_value=B,trainable=trainable_B,name='B_vector')
        self.use_em=use_em
        self.use_eh = use_eh
        # weights for computing input state (e_x will be built at build time when we know input dim)
        if self.use_em:
            # weight for squashing prior memory
            self.e_m = self.add_weight(shape=(order,1),initializer='glorot_uniform',trainable=True,name='mem_emb')
        if self.use_eh:
            # weight for squashing prior hidden state
            self.e_h =self.add_weight(shape=(hidden_dim,1),initializer='glorot_uniform',trainable=True,name='hidden_emb')
        # weight for computing portions of hidden states
        # W_h, W_m (W_x will be built at build time, when we know the input dim)
        self.W_h = self.add_weight(shape=(hidden_dim,hidden_dim),initializer='glorot_uniform',
                                   trainable=True,name='hidden_weights')
        self.W_m = self.add_weight(shape=(hidden_dim,order),initializer='glorot_uniform',
                                   trainable=True, name='memory_weights')

    def build(self, input_shape):
        # need to create W_x and e_x here
        self.W_x = self.add_weight(shape=(input_shape[-1],self.hidden_dim),initializer='glorot_uniform',
                                   trainable=True, name='input_weights')
        self.e_x = self.add_weight(shape=(input_shape[-1],1),initializer='glorot_uniform'
                                   ,trainable=True,name='input_emb')
        # shape is batch x timestep x H x W
        self.timesteps = input_shape[-3]

    def call(self,inputs):
        # need to compute memories for each timestep
        shape = tf.shape(inputs)
        batch = shape[0]
        timestep = shape[1]
        memory = tf.zeros((batch,self.order,1),dtype=tf.float32)
        # writing dummy timestep for t=-1
        hiddens =tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True,clear_after_read=False)
        hiddens = hiddens.write(0,tf.zeros((1,batch,self.hidden_dim),dtype=tf.float32))
        # iterating over the time dimension
        for i in range(1,self.timesteps+1):
            # transpose to get (batch,hidden_dim,1)
            last_hidden = tf.transpose(hiddens.read(i-1),perm=[1,2,0])
            feature = inputs[:,i,:]
            # computing u
            u = tf.matmul(feature,self.e_x)
            if self.use_eh:
                u = tf.add(u,tf.matmul(last_hidden,self.e_h))
            if self.use_em:
                u = tf.add(u,tf.matmul(memory,self.e_m))
            # computing the new memory (m_t)
            # need to expand B along the batch axis for scalar multiplication
            # need to match u to B for scalar batch element wise multiplication
            # U is of shape (batch,1) so we need to reshape it to (batch,order,1)
            u = tf.expand_dims(u,axis=1)
            u = tf.repeat(u,self.order,axis=1)
            expanded_B = tf.repeat(tf.expand_dims(self.B,axis=0),batch,axis=0)
            memory = tf.add(tf.matmul(self.A,memory),tf.multiply(expanded_B,u))
            # computing the new hidden state (h_t)
            in_act = tf.matmul(feature,self.W_x)
            in_act = tf.expand_dims(in_act,axis=2)
            hidden_act = tf.matmul(self.W_h,last_hidden)
            memory_act = tf.matmul(self.W_m,memory)
            # update hidden state for this timestep
            hidden = tf.nn.leaky_relu(tf.add(in_act,tf.add(hidden_act,memory_act))) #batch,hidden_dim,1
            # transpose to get (batch,1,hidden_dim)
            hiddens = hiddens.write(i,tf.transpose(hidden,perm=[2,0,1]))
        # concatenating along time dimension to get (time,batch,hidden_dim)
        hiddens = hiddens.concat()
        # removing the first timestep feature, since it was a dummy of zero values
        hiddens = hiddens[1:,:,:]
        # tranpose to get (batch,time,hidden_dim)
        hiddens = tf.transpose(hiddens,perm=[1,0,2])
        return hiddens



def get_state_space_matrices(order,theta):
    Q = np.arange(order, dtype=np.float64)
    R = (2 * Q + 1)[:, None] / theta
    j, i = np.meshgrid(Q, Q)

    A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
    B = (-1.0) ** Q[:, None] * R
    C = np.ones((1, order))
    D = np.zeros((1,))

    A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")
    return A,B

def build_H_parallel(A,B,sequence_length):
    print('A shape: ' + str(A.shape))
    print('B shape: '+ str(B.shape))
    # H is the matrix [A^0B, A^1B, ... , A^nB]
    H = []
    mult = tf.eye(A.shape[0])
    for i in range(sequence_length):
        to_add = tf.matmul(mult,B)
        mult = tf.matmul(mult,A)
        H.append(to_add)
    # concatenating so we have a d x sequence_length vector
    H = tf.concat(H,axis=-1)
    # getting fft for the fast multiplication provided in the paper
    print('H shape'+ str(H.shape))
    fft_H = tf.signal.rfft(H,fft_length=tf.convert_to_tensor([2*sequence_length],dtype=tf.int32))
    fft_H = tf.cast(fft_H,dtype=tf.complex64)
    # reshape to support batch multiplication
    fft_H = tf.reshape(fft_H,shape=(1,A.shape[0],24))
    print('fft_H shape: ' + str(fft_H.shape))
    return H,fft_H

def add_lmu(input,fft_H,hidden_dim):
    x = tf.keras.layers.Dense(1, activation='leaky_relu')(input)
    #print(x.shape)
    x = tf.keras.layers.Permute((2,1),input_shape=(23,1))(x)
    # fft on the input embedding
    x = tf.keras.layers.Lambda(lambda p: tf.signal.rfft(p, fft_length=tf.convert_to_tensor([2*23],dtype=tf.int32)))(x)
    # remove the pad element
    #x = tf.keras.layers.Lambda(lambda p: p[:,:,:-1])(x)
    x = tf.keras.layers.Lambda(lambda p: p*fft_H)(x)
    #x = tf.keras.layers.Multiply()([x,fft_H])
    #x = tf.keras.layers.Lambda(lambda p: tf.matmul(p,fft_H))(x)
    x = tf.keras.layers.Lambda(lambda p: tf.signal.irfft(p, fft_length=tf.convert_to_tensor([24],dtype=tf.int32)))(x)
    # remove padded timestep
    x=tf.keras.layers.Lambda(lambda p: p[:,:,:-1])(x)
    x = tf.keras.layers.Permute((2,1),input_shape=(10,23))(x)
    x = tf.keras.layers.Concatenate(axis=-1)([input, x])
    x = tf.keras.layers.Dense(hidden_dim, activation='leaky_relu')(x)
    # only want the last hidden output, not all of them
    #x = tf.keras.layers.Lambda(lambda p: p[:,-1,:])(x)
    return x



# this LMU consists of 3 lmu units of hidden_dim followed by a dense output layer with sigmoid binary output
def build_lmu(order,theta,hidden_dim,num_lmus=1):
    x=input = tf.keras.Input(shape=(23,2508))
    # reshaping input to flatten HxW dimensions
    x = tf.keras.layers.Reshape(target_shape=(23, 2508))(input)
    for i in range(num_lmus):
       x = KerasLMU(order,theta,hidden_dim)(x)

    # obtaining only the last hidden layer output
    x = tf.keras.layers.Lambda(lambda x: x[:,-1])(x)
    output = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(input, output)
    # Finally, we compute the cross-entropy loss between true labels and predicted labels to account for
    # the class imbalance between seizure and non-seizure depicting data
    # loss_func = keras.losses.categorical_crossentropy
    loss_func = tf.keras.losses.BinaryCrossentropy()
    optim = tf.keras.optimizers.RMSprop(learning_rate=0.001)
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

# this LMU consists of 3 lmu units of hidden_dim followed by a dense output layer with sigmoid binary output
def build_parallel_lmu(order,theta,hidden_dim,num_lmus=1):
    # building A and B matrices based on ODE  (this part is based on LMU example code from nengo)
    A,B = get_state_space_matrices(order,theta)
    # creating H and FFT(H) matrices from paper
    # sequence length of 23
    H,fft_H = build_H_parallel(A,B,23)

    fft_H = K.constant(fft_H,dtype='complex64')


    input = tf.keras.Input(shape=(23,2508))
    # reshaping input to flatten HxW dimensions
    x = tf.keras.layers.Reshape(target_shape=(23, 2508))(input)
    for i in range(num_lmus):
       x = add_lmu(x,fft_H,hidden_dim)
    # output classfication
    # flatten the concatenated hidden states
    x = tf.keras.layers.Flatten()(x)
    # actually, we only want the last hidden state
    #x = tf.keras.layers.Lambda(lambda p: p[:,-1,:])(x)
    #x = tf.keras.layers.Dense(1024)(x)
    #x = tf.keras.layers.Dense(256)(x)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(x)

    model = tf.keras.Model(input, output)
    # Finally, we compute the cross-entropy loss between true labels and predicted labels to account for
    # the class imbalance between seizure and non-seizure depicting data
    # loss_func = keras.losses.categorical_crossentropy
    loss_func = tf.keras.losses.binary_crossentropy
    optim = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
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
    model.compile(loss=loss_func, optimizer=optim, metrics=metrics, )
    print('Parallel LMU model successfully built')
    return model

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


#converted = nengo_dl.Converter(model)
'''

#model = build_lmu(256,784,256,num_lmus=2)