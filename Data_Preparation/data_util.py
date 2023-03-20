import os

import tensorflow as tf
from pathlib import Path
import numpy as np
from constants import ROOT_DIR

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
        labels = tf.zeros(shape = (data.shape[0]),dtype=tf.uint8)
    else:
        #labels = np.ones(shape = (data.shape[0]),dtype = np.int8)
        labels = tf.ones(shape=(data.shape[0]), dtype=tf.uint8)

    #print(data[0].shape)
    #print(data)
    data = tf.convert_to_tensor(data,dtype=tf.float32)

    return data,labels

# applying stft to a tf tensor to convert to frequency domain
def stft_samples(tf_tensor):
    tensor_stft = tf.signal.stft(tf_tensor, frame_length=256, frame_step=128, fft_length=256, pad_end=False)
    indices_wanted = [i for i in range(1, 57)] + [i for i in range(64, 117)] + [i for i in range(124,129)]
    # filtering out bad frequencies
    # removing DC component (0 Hz), the 57-63Hz and 117-123Hz bands (specific for chb-mit dataset)
    tensor_stft = tf.gather(tensor_stft, axis=-1, indices=indices_wanted)
    # obtaining magnitude, since stft returns complex numbers
    tensor_stft = tf.abs(tensor_stft)
    return tensor_stft

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
        # need to grab data and filter out any data relating to the leave out patient
        dataset_glob_path = os.path.join(ROOT_DIR,'Data_Preparation','Processed_Data','*'
                                         ,str(window_size)+'-'+'*'+split+'.npy')
        dataset = tf.data.Dataset.list_files(dataset_glob_path)
        #dataset = tf.data.Dataset.list_files(ROOT_DIR'./Processed_Data/*/'+str(window_size)+'-'+'*'+split+'.npy')
        dataset_list = list(dataset)
        filtered_list = []
        for data in dataset_list:
            raw = data.numpy()
            if leave_out not in str(raw):
                filtered_list.append(raw)
        filtered_list = np.array(filtered_list)
        dataset = tf.data.Dataset.from_tensor_slices(filtered_list)
    else:
        # we want the entire data for patient chb01 in this case for testing
        dataset_glob_path = os.path.join(ROOT_DIR, 'Data_Preparation', 'Processed_Data', leave_out
                                         , str(window_size) + '-*.npy')
        dataset = tf.data.Dataset.list_files(dataset_glob_path)
        #dataset = tf.data.Dataset.list_files('./Processed_Data/'+leave_out+'/'+str(window_size)+'*.npy')
    # converting filename to batched numpy array and batched label tensors
    dataset = dataset.map(lambda x: tf.py_function(npy_to_tf,inp=[x],Tout=[tf.float32,tf.uint8]),
                          num_parallel_calls=4)
    # taking the batched data and batched labels and flattening them (preserves order)
    examples = dataset.flat_map(lambda example,label: tf.data.Dataset.from_tensor_slices(example))
    # applying stft transform to our examples
    examples = examples.map(stft_samples,num_parallel_calls=4)
    # need to squeeze the extra dimension if needed
    examples = examples.map(tf.squeeze,num_parallel_calls=4)
    # if our window size is 1, we need to explicitly set a channel to process like an image
    if window_size == 1:
        examples= examples.map(lambda example: tf.expand_dims(example,axis=0),num_parallel_calls=4)
    labels = dataset.flat_map(lambda example,label: tf.data.Dataset.from_tensor_slices(label))
    # need to provide labels as a one-hot-encoding in keras api
    #labels = labels.map(lambda label: tf.one_hot(label,depth=2),num_parallel_calls=tf.data.AUTOTUNE)
    # zipping together flattened examples to form final dataset
    dataset = tf.data.Dataset.zip((examples,labels))
    # adding sample weighting for imbalances only for the training set
    if split=='train' :
        # getting imbalance count
        num_interictal, num_ictal = get_class_counts(dataset)
        imbalance = -(num_interictal//-num_ictal)
        ictals = dataset.filter(lambda example,label: label==1)
        ictals = ictals.map(lambda example,label: (example,label,imbalance))
        interictals = dataset.filter(lambda example,label: label==0)
        interictals=interictals.map(lambda example,label: (example,label,1))
        dataset = ictals.concatenate(interictals)
        #dataset = dataset.map(lambda example,label: add_sample_weighting(example,imbalance))
    return dataset

# function that gives class counts for a binary classification task
# input dataset is of the form (example,label)
# labels are 0 or 1
# we return the count of zeros first, then the count of ones
def get_class_counts(tf_dataset):
    total = tf_dataset.reduce(0,lambda y,_: y+1)
    ones = tf_dataset.filter(lambda example,label: label==1).reduce(0,lambda y,_:y+1)
    zeros = total-ones
    return zeros,ones


# this gets train,val,test splits on the PATIENT-level
# i.e, if we specify 'chb01' we get first 80% as train, consecutive 10% as val, and latter 10% as test for the specific patient chb01
def get_patient_level_data(window_size=1,patient='chb01'):
    # grabbing files related to this patient
    train_glob_path = os.path.join(ROOT_DIR, 'Data_Preparation', 'Processed_Data', patient
                                     , str(window_size) + '-*train.npy')
    val_glob_path = os.path.join(ROOT_DIR, 'Data_Preparation', 'Processed_Data', patient
                                     , str(window_size) + '-*val.npy')

    # train_dataset contains files for the first 75% of examples
    train_dataset = tf.data.Dataset.list_files(train_glob_path,shuffle=False)
    # val_dataset contains files for the latter 25% of examples from a different time period
    val_dataset = tf.data.Dataset.list_files(val_glob_path,shuffle=False)

    # converting filename to batched numpy array and batched label tensors
    train_dataset = train_dataset.map(lambda x: tf.py_function(npy_to_tf, inp=[x], Tout=[tf.float32, tf.uint8]),
                          num_parallel_calls=4)
    val_dataset = val_dataset.map(lambda x: tf.py_function(npy_to_tf, inp=[x], Tout=[tf.float32, tf.uint8]),
                                      num_parallel_calls=4)
    # taking the batched data and batched labels and flattening them (preserves order)
    train_examples = train_dataset.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(example))
    val_examples = val_dataset.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(example))
    # applying stft transform to our examples
    train_examples = train_examples.map(stft_samples, num_parallel_calls=4)
    val_examples = val_examples.map(stft_samples, num_parallel_calls=4)
    # need to squeeze the extra dimension if needed
    train_examples = train_examples.map(tf.squeeze, num_parallel_calls=4)
    val_examples = val_examples.map(tf.squeeze, num_parallel_calls=4)
    # if our window size is 1, we need to explicitly set a channel to process like an image
    if window_size == 1:
        train_examples = train_examples.map(lambda example: tf.expand_dims(example, axis=0), num_parallel_calls=4)
        val_examples = val_examples.map(lambda example: tf.expand_dims(example, axis=0), num_parallel_calls=4)
    train_labels = train_dataset.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(label))
    val_labels = train_dataset.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(label))

    # need to provide labels as a one-hot-encoding in keras api
    # labels = labels.map(lambda label: tf.one_hot(label,depth=2),num_parallel_calls=tf.data.AUTOTUNE)
    # zipping together flattened examples to form final dataset
    train = tf.data.Dataset.zip((train_examples, train_labels))
    val = tf.data.Dataset.zip((val_examples, val_labels))

    total_count = train.reduce(0,lambda y,_:y+1)
    total_count +=val.reduce(0,lambda y,_:y+1)

    # train dataset contains 75% of examples
    # val contains 25% of examples
    # firstly, we will transfer 5% of the total as interictal data to the train_dataset
    interictal_val = val.filter(lambda example,label: label==0)
    ictal_val = val.filter(lambda example,label: label==1)
    val_interictal_count = interictal_val.reduce(0,lambda y,_:y+1)
    val_ictal_count  = ictal_val.reduce(0,lambda y,_:y+1)

    train = train.concatenate(interictal_val.take(int(0.05 * total_count.numpy())))
    interictal_val = interictal_val.skip(int(0.05*total_count.numpy()))
    val_interictal_count -= int(0.05 * total_count.numpy())

    # splitting ictal and interictal now in half and that will become our val and test sets
    val = ictal_val.take(int(0.5*val_ictal_count.numpy()))
    val = val.concatenate(interictal_val.take(int(0.5*val_interictal_count.numpy())))
    ictal_val = ictal_val.skip(int(0.5*val_ictal_count.numpy()))
    interictal_val = interictal_val.skip(int(0.5*val_interictal_count.numpy()))

    ictal_remaining = (val_ictal_count-int(0.5*val_ictal_count.numpy())).numpy()
    interictal_remaining = (val_interictal_count - int(0.5*val_interictal_count.numpy())).numpy()
    test = ictal_val.take(ictal_remaining).concatenate(interictal_val.take(interictal_remaining))

    # adding sample weighting for imbalances only for the training set
    # getting imbalance count
    num_interictal, num_ictal = get_class_counts(train)
    imbalance = -(num_interictal // -num_ictal)
    ictals = train.filter(lambda example, label: label == 1)
    ictals = ictals.map(lambda example, label: (example, label, imbalance))
    interictals = train.filter(lambda example, label: label == 0)
    interictals = interictals.map(lambda example, label: (example, label, 1))
    train = ictals.concatenate(interictals)
    return train,val,test

#train,val,test = get_patient_level_data()
#print(len(list(iter(val.filter(lambda example,label:label==1)))))
#print(len(list(iter(train.filter(lambda example,label,weight:label==1)))))
#print(len(list(iter(test.filter(lambda example,label:label==1)))))
