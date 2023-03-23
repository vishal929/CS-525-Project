import os

import tensorflow as tf
from pathlib import Path
import numpy as np
from constants import ROOT_DIR

# this file houses logic associated with data preparation for deep learning
# i.e loading the dataset, and computing class weights

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


# getting the number of seizures for a specific patient
# this is a helper method to picking seizures to use for one-leave-out-validation
def get_num_seizures(patient='chb01'):
    glob_res = Path.glob(Path(ROOT_DIR,'Data_Preparation','Processed_Data',patient),'1-*-ictal_train.npy')
    return len(list(glob_res))

# for each patient, we take val and train and leave out 1 seizure
# we repeat this 5 times and then average for a score
def get_seizure_leave_out_data(seizure_number,window_size=1,patient='chb01'):
    # grabbing seizure and non-seizure files for the patient
    # grabbing files related to this patient
    ictal_test_glob_path =os.path.join(ROOT_DIR, 'Data_Preparation', 'Processed_Data', patient
                                            , str(window_size) + '-'+str(seizure_number)+'-ictal_*.npy')
    train_glob_path = os.path.join(ROOT_DIR, 'Data_Preparation', 'Processed_Data', patient
                                            , str(window_size) + '-*_train.npy')
    val_glob_path = os.path.join(ROOT_DIR, 'Data_Preparation', 'Processed_Data', patient
                                            , str(window_size) + '-*_val.npy')

    test = tf.data.Dataset.list_files(ictal_test_glob_path)
    train = tf.data.Dataset.list_files(train_glob_path)
    val = tf.data.Dataset.list_files(val_glob_path)

    # removing any files that are in train or val that are also in test_seizure (for leave out)
    test_set = set(iter(test.as_numpy_iterator()))
    train_list = list(iter(train.as_numpy_iterator()))
    val_list = list(iter(val.as_numpy_iterator()))
    filtered_train_list = []
    filtered_val_list = []
    for file in train_list:
        if file not in test_set:
            filtered_train_list.append(file)
    for file in val_list:
        if file not in test_set:
            filtered_val_list.append(file)
    train_list = filtered_train_list
    val_list = filtered_val_list
    train = tf.data.Dataset.from_tensor_slices(train_list)
    val = tf.data.Dataset.from_tensor_slices(val_list)

    # converting filename to batched numpy array and batched label tensors
    train = train.map(lambda x: tf.py_function(npy_to_tf, inp=[x], Tout=[tf.float32, tf.uint8]),
                                      num_parallel_calls=4)
    val = val.map(lambda x: tf.py_function(npy_to_tf, inp=[x], Tout=[tf.float32, tf.uint8]),
                                  num_parallel_calls=4)
    test = test.map(lambda x: tf.py_function(npy_to_tf, inp=[x], Tout=[tf.float32, tf.uint8]),
                                  num_parallel_calls=4)
    # taking the batched data and batched labels and flattening them (preserves order)
    train_examples = train.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(example))
    val_examples = val.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(example))
    test_examples = test.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(example))
    # applying stft transform to our examples
    train_examples = train_examples.map(stft_samples, num_parallel_calls=4)
    val_examples = val_examples.map(stft_samples, num_parallel_calls=4)
    test_examples = test_examples.map(stft_samples, num_parallel_calls=4)
    # need to squeeze the extra dimension if needed
    train_examples = train_examples.map(tf.squeeze, num_parallel_calls=4)
    val_examples = val_examples.map(tf.squeeze, num_parallel_calls=4)
    test_examples = test_examples.map(tf.squeeze, num_parallel_calls=4)
    # if our window size is 1, we need to explicitly set a channel to process like an image
    if window_size == 1:
        train_examples = train_examples.map(lambda example: tf.expand_dims(example, axis=0), num_parallel_calls=4)
        val_examples = val_examples.map(lambda example: tf.expand_dims(example, axis=0), num_parallel_calls=4)
        test_examples = test_examples.map(lambda example: tf.expand_dims(example, axis=0), num_parallel_calls=4)
    train_labels = train.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(label))
    val_labels = val.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(label))
    test_labels = test.flat_map(lambda example, label: tf.data.Dataset.from_tensor_slices(label))

    # need to provide labels as a one-hot-encoding in keras api
    # labels = labels.map(lambda label: tf.one_hot(label,depth=2),num_parallel_calls=tf.data.AUTOTUNE)
    # zipping together flattened examples to form final dataset
    train = tf.data.Dataset.zip((train_examples, train_labels))
    val = tf.data.Dataset.zip((val_examples, val_labels))
    test = tf.data.Dataset.zip((test_examples,test_labels))

    # applying sample weighting to train data
    num_interictal, num_ictal = get_class_counts(train)
    imbalance = -(num_interictal // -num_ictal)
    ictals = train.filter(lambda example, label: label == 1)
    ictals = ictals.map(lambda example, label: (example, label, imbalance))
    interictals = train.filter(lambda example, label: label == 0)
    interictals = interictals.map(lambda example, label: (example, label, 1))
    train = ictals.concatenate(interictals)

    return train,val,test

'''
train,val,test = get_seizure_leave_out_data(1)
print(train.reduce(0,lambda y,_: y+1))
print(val.reduce(0,lambda y,_: y+1))
print(test.reduce(0,lambda y,_: y+1))

print(train.filter(lambda example,label,weight: label==1).reduce(0,lambda y,_:y+1))
print(val.filter(lambda example,label: label==1).reduce(0,lambda y,_:y+1))
print(test.filter(lambda example,label: label==1).reduce(0,lambda y,_:y+1))
'''