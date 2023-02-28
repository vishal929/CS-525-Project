from pathlib import Path
import numpy as np

# this file houses logic associated with data preparation for deep learning
# i.e loading the dataset, and computing class weights

# this function looks at each patients train and validation data for the processed directory
# we index each example so that at train time or test time, we can load examples just with their indices
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

# tensorflow data pipeline based on our metadata


print(grab_processed_metadata()[0][5000])