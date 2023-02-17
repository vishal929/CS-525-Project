import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path

# chb-mit includes 23 channels sampled at 256hz

# function that gets summary files, reads them, and gets seizure timestamps
def get_seizure_timestamps():
    pass

# function that searches the data directory for filenames globbed as *.edf (all our data files for the eeg recordings)
def get_eegs():
    eeg_paths = Path.glob(Path('./Data'),'*/*.edf')
    #seizure_paths = Path.glob(Path('./Data'),'*/*.seizures')
    path_list = list(eeg_paths) #+ list(seizure_paths)
    return [str(path) for path in path_list]

# function that takes an eeg recording file and splits it into windows based on a window size and sampling frequency
# NOTE: window size is in seconds-> number of samples will be taken care of automagically by this function
def window_recordings(file_path, window_size=12):
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
        window = eeg_raw_data[:,curr_window:curr_window+(sampling_frequency*window_size)]
        # we want to break if the window is less than our desired window size by splicing
        if window.shape[1]<sampling_frequency*window_size:
            break
        #print(window.shape)
        new_data.append(window)
        # moving the window by just 1 second
        curr_window += sampling_frequency

    # we want to return something like (window,num_channels,samples)
    return np.array(new_data)

# function that takes an eeg recording, and applies the short term fourier transform with window and overlap parameters
# we filter out noise frequency and DC frequency specific to the CHB-MIT scalp EEG dataset
def stft_recordings(eeg_data,window=256,overlap=None):
    # applying stft to data
    frequencies = mne.time_frequency.stft(eeg_data,wsize=window,tstep=overlap)
    # removing DC component (0 Hz), the 57-63Hz and 117-123Hz bands (specific for
    frequencies = np.delete(frequencies, [0, *[i for i in range(57, 64, 1)], *[i for i in range(117, 124, 1)]], axis=1)
    return frequencies

'''
    Below is some basic logic I wrote while testing stuff
'''

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


