# this model is a recurrent model based on LMUs for processing 12s window segments for binary classification
# since we do not require intermediate outputs, we can parallelize the lmu!

import nengo
import nengo_dl
#from nengo.utils.filter_design import cont2discrete
from scipy.signal import cont2discrete
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K


# TEMP FIX FOR SOME RUNTIME ISSUES
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    mult = tf.identity(A)
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
    x = tf.keras.layers.Dense(1, activation='relu')(input)
    x = tf.keras.layers.Reshape(target_shape=(1,23))(x)
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
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    # only want the last hidden output, not all of them
    #x = tf.keras.layers.Lambda(lambda p: p[:,-1,:])(x)
    return x

# this LMU consists of 3 lmu units of hidden_dim followed by a dense output layer with sigmoid binary output
def build_parallel_lmu(order,theta,hidden_dim,num_lmus=1):
    # building A and B matrices based on ODE  (this part is based on LMU example code from nengo)
    A,B = get_state_space_matrices(order,theta)
    # creating H and FFT(H) matrices from paper
    # sequence length of 23
    H,fft_H = build_H_parallel(A,B,23)

    fft_H = K.constant(fft_H,dtype='complex64')

    input = tf.keras.Input(shape=(23,22,114))
    # reshaping input to flatten HxW dimensions
    x = tf.keras.layers.Reshape(target_shape=(23, 2508))(input)
    for i in range(num_lmus):
       x = add_lmu(x,fft_H,hidden_dim)
    # output classfication
    # flatten the concatenated hidden states
    #x = tf.keras.layers.Flatten()(x)
    # actually, we only want the last hidden state
    x = tf.keras.layers.Lambda(lambda p: p[:,-1,:])(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dense(256)(x)
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

model = build_parallel_lmu(40,50,140,num_lmus=3)
print(model.predict(tf.random.normal(shape=(1,23,22,114),dtype=tf.float32),batch_size=1))


trainable_count = int(
    sum(K.count_params(layer) for layer in model.trainable_weights))
non_trainable_count = int(
    sum(K.count_params(layer) for layer in model.non_trainable_weights))

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))