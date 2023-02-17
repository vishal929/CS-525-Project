import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from os import path
import pickle


# chb-mit includes 23 channels sampled at 256hz

# parent function to preprocess data in the Data directory for some given window size
def preprocess_data(window_size=12):
    # checking if summary pickle exists
    if Path.exists(Path('./Processed_Data/summary.pickle')):
        # load the pickle
        pickle_file = open(Path('./Processed_Data/summary.pickle'),'rb')
        final_mapping = pickle.load(pickle_file)
        pickle_file.close()
        print("LOADED A PICKLE FILE OF PROCESSED SUMMARIES!")
        print(final_mapping)
    else:
        # getting the summary data
        summary_files = get_summary_files()
        # getting dictionary mapping for every seizure
        final_mapping = {}
        for summary_file in summary_files:
            final_mapping.update(get_seizure_timestamps(summary_file))
        # combining dictionaries into 1 and then we can keep this object as a pickle object
        # writing the pickle to filesystem
        pickle_file = open(Path('./Processed_Data/summary.pickle'),'wb')
        pickle.dump(final_mapping,pickle_file)
        pickle_file.close()
        print("WROTE A PICKLE OF PROCESSED SUMMARIES!")
    # getting the eeg data files
    eeg_files = get_eegs()
    # we only want to process eeg data one summary file at a time
    # we want to keep them together so we have it organized per patient
    # TODO: file processing + tagging (ideally add tagging to the window_recordings() function)



# function that retrieves all summary text files
def get_summary_files():
    file_paths = Path.glob(Path('./Data'),'*/*-summary.txt')
    path_list = list(file_paths)
    return [str(path) for path in path_list]

# function that gets seizure timestamps from a summary file
# records should be "filename: [(start_time,end_time),(start_time_2,end_time_2),...]"
def get_seizure_timestamps(summary_filepath):
    # reading in the summary file
    f = open(summary_filepath, 'r')
    lines = f.readlines()
    f.close()
    # filtering lines we dont need
    seizure_mapping = {}
    curr_filename = None
    for idx,line in enumerate(lines):
        line = line.lower()
        if line.startswith('file name'):
            # start a new record
            curr_filename = line.strip().split(':')[1].strip()
            seizure_mapping[curr_filename] = []
        if line.startswith('seizure start time'):
            # the next line will be the seizure end time
            seizure_start_time = int(line.split(':')[1].split()[0])
            seizure_end_time = int(lines[idx+1].split(':')[1].split()[0])
            # adding this data to the dictionary entry
            seizure_mapping[curr_filename].append((seizure_start_time,seizure_end_time))
    return seizure_mapping


# function that searches the data directory for filenames globbed as *.edf (all our data files for the eeg recordings)
def get_eegs():
    eeg_paths = Path.glob(Path('./Data'), '*/*.edf')
    # seizure_paths = Path.glob(Path('./Data'),'*/*.seizures')
    path_list = list(eeg_paths)  # + list(seizure_paths)
    return [str(path) for path in path_list]


# function that takes an eeg recording file and splits it into windows based on a window size and sampling frequency
# NOTE: window size is in seconds-> number of samples will be taken care of automagically by this function
# seizure times is the list of tuples [(start_time,end_time),(start_time_2,end_time_2),...]
# TODO: implement tagging here
def window_recordings(file_path, seizure_times, window_size=12):
    # reading the eeg file
    eeg_raw = mne.io.read_raw_edf(file_path)

    # removing the redundant channel
    eeg_raw = eeg_raw.drop_channels(['T8-P8-1']).rename_channels({'T8-P8-0': 'T8-P8'})

    sampling_frequency = int(eeg_raw.info['sfreq'])

    # getting data in the shape (num_channels,samples) where we sample 'sampling_frequency * seconds'
    eeg_raw_data = eeg_raw.get_data()

    # windowing based on window_size and sampling frequency
    total_samples = eeg_raw_data.shape[1]
    curr_window = 0
    new_data = []
    while curr_window != total_samples:
        window = eeg_raw_data[:, curr_window:curr_window + (sampling_frequency * window_size)]
        # we want to break if the window is less than our desired window size by splicing
        if window.shape[1] < sampling_frequency * window_size:
            break
        # print(window.shape)
        new_data.append(window)
        # moving the window by just 1 second
        curr_window += sampling_frequency

    # we want to return something like (window,num_channels,samples),[labels for each window]
    return np.array(new_data)


# function that takes an eeg recording, and applies the short term fourier transform with window and overlap parameters
# we filter out noise frequency and DC frequency specific to the CHB-MIT scalp EEG dataset
def stft_recordings(eeg_data, window=256, overlap=None):
    # applying stft to data
    frequencies = mne.time_frequency.stft(eeg_data, wsize=window, tstep=overlap)
    # removing DC component (0 Hz), the 57-63Hz and 117-123Hz bands (specific for
    frequencies = np.delete(frequencies, [0, *[i for i in range(57, 64, 1)], *[i for i in range(117, 124, 1)]], axis=1)
    return frequencies


'''
    Below is some basic logic I wrote while testing stuff
'''

#preprocess_data()

'''
# read in an edf file
test = mne.io.read_raw_edf('./Data/chb01/chb01_01.edf')
print(test.info)
print(test.info['ch_names'])

# WE HAVE A DUPLICATE CHANNEL!
# it seems this is why the research paper mentions 22 channels but the info specifies 23 ...
# why do they do this :(
print(np.array_equal(test['T8-P8-1'][0],test['T8-P8-0'][0],equal_nan=True))
print(np.array_equal(test['T8-P8-1'][1],test['T8-P8-0'][1],equal_nan=True))

# dropping a channel from test
test = test.drop_channels(['T8-P8-1']).rename_channels({'T8-P8-0': 'T8-P8'})
print(test.info['ch_names'])

#test.plot(duration=5, n_channels=23,show=False)
#plt.show()

test_data = test.get_data()

# we have num_channels x samples
# we have a sampling rate of 256 samples a second
# shape is (22,921600) for chb01_01.edf
print(test_data.shape)

# getting 12 seconds of data based on sampling rate of 256 samples a second
test_data = test_data[:,:3072]
print(test_data.shape)

# short time fast fourier transform
test_fft = mne.time_frequency.stft(test_data,wsize=256)
# (22,129,7200)
# 22 channels, 129 frequencies (0 to 128), and 7200 timesteps by shifting half a window each time
print(test_fft.shape)

# removing DC component (0 Hz), the 57-63Hz and 117-123Hz bands
test_fft = np.delete(test_fft,[0,*[i for i in range(57,64,1)],*[i for i in range(117,124,1)]],axis=1)

print(test_fft.shape)
'''
