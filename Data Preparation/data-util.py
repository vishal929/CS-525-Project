import tensorflow as tf
from pathlib import Path
import numpy as np

# this file houses logic associated with data preparation for deep learning
# i.e loading the dataset, and computing class weights

# this function looks at each patients train and validation data for the processed directory
# we index each example so that at train time or test time, we can load examples just with their indices
'''DONT USE THE BELOW FUNCTION -> I Just thought i'd leave it here in case something goes wrong with tf data api in future'''
def grab_processed_metadata(window_size=1):
    # finding the processed files
    ictal_train = Path.glob(Path('./Processed_Data'),'*/'+str(window_size)+'-ictal_train.npy')
    ictal_val = Path.glob(Path('./Processed_Data'), '*/' + str(window_size) + '-ictal_val.npy')
    interictal_train = Path.glob(Path('./Processed_Data'), '*/' + str(window_size) + '-interictal_train.npy')
    interictal_val = Path.glob(Path('./Processed_Data'), '*/' + str(window_size) + '-interictal_val.npy')

    # mapping by dirname (i.e) chb01
    # want something like train = {index: (filepath,augmented_index)}
    # filepath will implicitly encode the label (i.e interictal_train or ictal_train)
    # augmented index is the index of the example in the npy file while index is the global dataset index
    # we can do the same thing for val
    # we can index validation set differently from the train set
    train_examples = list(ictal_train) + list(interictal_train)
    val_examples = list(ictal_val) + list(interictal_val)
    train_count = 0
    val_count = 0
    train_map = {}
    val_map = {}
    for file in train_examples:
        data = np.load(file)
        # counting number of examples
        file_count = data.shape[0]
        for i in range(train_count,train_count+file_count):
            train_map[i] = (file,i-train_count)
        train_count += file_count
        del data

    for file in val_examples:
        data = np.load(file)
        # counting number of examples
        file_count = data.shape[0]
        for i in range(val_count, val_count + file_count):
            val_map[i] = (file, i - val_count)
        val_count += file_count
        del data

    return train_map,val_map

# loading numpy file data and labels
# 0 will be for interictal, 1 will be for ictal
def npy_to_tf(filename):
    filename_str = filename.numpy()
    #print(filename_str)
    data = np.load(filename_str)
    if tf.strings.regex_full_match(filename,'.*interictal.*'):
        #labels = np.zeros(shape = (data.shape[0]))
        labels = tf.zeros(shape = (data.shape[0]),dtype=tf.float32)
    else:
        #labels = np.ones(shape = (data.shape[0]),dtype = np.int8)
        labels = tf.ones(shape=(data.shape[0]), dtype=tf.float32)

    #print(data[0].shape)
    #print(data)
    data = tf.convert_to_tensor(data,dtype=tf.float32)

    return data,labels


''' USE THIS FUNCTION TO GET A DATASET WE CAN LOAD INTO OUR KERAS MODEL'''
# tensorflow data pipeline based on our metadata
# we specify a split (either 'train','val','test')
# we specify a window size
# we specify a patient to leave out i.e if we specify 'chb01' then patient 1 is not included in train or val
# # this patient will be included in the test set
def tf_dataset(split='train',window_size=1,leave_out='chb01'):
    split = split.strip().lower()
    if split not in {'test','val','train'}:
        raise Exception('split provided is not one of train,test, or val')
    if split != 'test':
        dataset = tf.data.Dataset.list_files('./Processed_Data/*/'+str(window_size)+'-'+'*'+split+'.npy')
    else:
        # we want the entire data for patient chb01 in this case for testing
        dataset = tf.data.Dataset.list_files('./Processed_Data/'+leave_out+'/'+str(window_size)+'*.npy')
    # converting filename to batched numpy array and batched label tensors
    dataset = dataset.map(lambda x: tf.py_function(npy_to_tf,inp=[x],Tout=[tf.float32,tf.float32]))
    # taking the batched data and batched labels and flattening them (preserves order)
    examples = dataset.flat_map(lambda example,label: tf.data.Dataset.from_tensor_slices(example))
    labels = dataset.flat_map(lambda example,label: tf.data.Dataset.from_tensor_slices(label))
    # zipping together flattened examples to form final dataset
    dataset = tf.data.Dataset.zip((examples,labels))
    return dataset

''' EXAMPLE OF HOW TO COUNT IN THE DATASET (we need eager execution due to transformations)'''
'''
test = tf_dataset(split='val')
count = test.reduce(0, lambda x,_: x+1).numpy()
print(count)
'''