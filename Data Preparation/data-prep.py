import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
import os
import pickle
import scipy.signal as signal
import datetime


# chb-mit includes 23 channels sampled at 256hz

# parent function to preprocess data in the Data directory for some given window size
def preprocess_data(window_size=12):
    # checking if summary pickle exists
    if Path.exists(Path('./Processed_Data/summary.pickle')):
        # load the pickle
        pickle_file = open(Path('./Processed_Data/summary.pickle'), 'rb')
        final_mapping = pickle.load(pickle_file)
        pickle_file.close()
        print("LOADED A PICKLE FILE OF PROCESSED SUMMARIES!")
        print(final_mapping)
    else:
        # getting the summary data
        summary_files = get_summary_files()
        # getting dictionary mapping for metadata for all files
        final_mapping = {}
        for summary_file in summary_files:
            # mapping should only have "chb01" not "chb01-summary.txt"
            record_list = get_seizure_timestamps(summary_file)
            if 'chb24' in summary_file:
                # need to fetch records that were not annotated in chb24 summary file
                record_list = grab_missing_records(record_list)
                # need to generate timestamps for record_list since these are not given
                record_list = generate_timesteps(record_list, 'chb24')
            final_mapping[os.path.basename(summary_file)[:5]] = record_list
            # final_mapping.update(get_seizure_timestamps(summary_file))
        # combining dictionaries into 1 and then we can keep this object as a pickle object
        # writing the pickle to filesystem
        pickle_file = open(Path('./Processed_Data/summary.pickle'), 'wb')
        pickle.dump(final_mapping, pickle_file)
        pickle_file.close()
        print("WROTE A PICKLE OF PROCESSED SUMMARIES!")

    # validating the mapping based on records file
    print('validating that all record metadata was grabbed ...')
    missing = validate_records(os.path.join('.', 'Data', 'RECORDS'), final_mapping)
    if len(missing) == 0:
        print('ALL RECORD METADATA VALIDATED AND SUCCESSFULLY GRABBED!')

    # getting the eeg data files
    eeg_files = get_eegs()
    # we only want to process eeg data one summary file at a time
    # we want to keep them together so we have it organized per patient
    # TODO: file processing + tagging (ideally add tagging to the window_recordings() function)
    return final_mapping


# function that retrieves all summary text files
def get_summary_files():
    file_paths = Path.glob(Path('./Data'), '*/*-summary.txt')
    path_list = list(file_paths)
    return [str(path) for path in path_list]


# function that gets seizure timestamps from a summary file and the associated time period
# records should be {filename: [(file_start,file_end),(seizure_start,seizure_end),(seizure_start_2,seizure_end_2),...]}
# we only care about absolute time (in seconds) for each patient
# for example, the first recording should have (0,seconds_taken)
# ... the second recording should have (start_in_seconds, start_in_seconds + seconds_taken)
# we return a list of records for this summary filepath
def get_seizure_timestamps(summary_filepath):
    # reading in the summary file
    f = open(summary_filepath, 'r')
    lines = f.readlines()
    f.close()
    # filtering lines we dont need
    total_files = []
    seizure_mapping = {}
    curr_filename = None
    last_end_time = None
    last_absolute_end_time = None
    for idx, line in enumerate(lines):
        line = line.lower()
        if line.startswith('file name'):
            # start a new record
            curr_filename = line.strip().split(':')[1].strip()
            # next 2 lines are file start and end time

            # getting hour, minutes, seconds
            file_start_elements = lines[idx + 1].split(':')
            file_start_elements = file_start_elements[len(file_start_elements) - 3:]

            file_end_elements = lines[idx + 2].split(':')
            file_end_elements = file_end_elements[len(file_end_elements) - 3:]

            # IMPORTANT: CHB24 has no file start and end times, so we will need to generate those
            if 'chb24' in curr_filename:
                start_seconds = None
                end_seconds = None
            else:
                s_hour, s_min, s_sec = [int(element) for element in file_start_elements]
                e_hour, e_min, e_sec = [int(element) for element in file_end_elements]

                # need to convert to seconds

                # regularization (some summary files go over 24, which python time package does not like)
                if s_hour >= 24:
                    s_hour -= 24
                if e_hour >= 24:
                    e_hour -= 24

                start_object = datetime.datetime.strptime(str(s_hour)+':'+str(s_min)+':'+str(s_sec),'%H:%M:%S')
                end_object = datetime.datetime.strptime(str(e_hour) + ':' + str(e_min) + ':' + str(e_sec), '%H:%M:%S')
                # need to account for the day to avoid negative differences
                while last_end_time is not None and start_object < last_end_time:
                    # need to increment the day of this object by 1 i.e we are going from something like 23 hr to 2 hr
                    start_object += datetime.timedelta(days=1)

                while end_object < start_object:
                    # need to account for wraparound, we can just increment the day of end_object by 1
                    end_object += datetime.timedelta(days=1)

                # getting timedelta in seconds
                seconds_elapsed = int((end_object-start_object).total_seconds())
                if last_end_time is None:
                    start_seconds = 0
                    end_seconds = seconds_elapsed
                else:
                    seconds_elapsed_between_files = int((start_object-last_end_time).total_seconds())
                    start_seconds = last_absolute_end_time + seconds_elapsed_between_files
                    end_seconds = start_seconds + seconds_elapsed
                last_end_time = end_object
                last_absolute_end_time = end_seconds


            if len(seizure_mapping) != 0:
                total_files.append(seizure_mapping)
                seizure_mapping = {}
            seizure_mapping[curr_filename] = [(start_seconds, end_seconds)]
        if line.startswith('seizure start time'):
            # the next line will be the seizure end time
            seizure_start_time = int(line.split(':')[1].split()[0])
            seizure_end_time = int(lines[idx + 1].split(':')[1].split()[0])
            # adding this data to the dictionary entry
            seizure_mapping[curr_filename].append((seizure_start_time, seizure_end_time))
    # appending the final filename to the total list
    total_files.append(seizure_mapping)
    return total_files


# function that searches the data directory for filenames globbed as *.edf (all our data files for the eeg recordings)
def get_eegs():
    eeg_paths = Path.glob(Path('./Data'), '*/*.edf')
    # seizure_paths = Path.glob(Path('./Data'),'*/*.seizures')
    path_list = list(eeg_paths)  # + list(seizure_paths)
    return [str(path) for path in path_list]

# reading eeg data specific to chb-mit dataset
def read_mne_data(filename):
    # reading the eeg file and excluding dummy '-' channels
    eeg_raw = mne.io.read_raw_edf(filename, exclude=['-'])

    # removing the redundant channel
    eeg_raw = eeg_raw.drop_channels(['T8-P8-1']).rename_channels({'T8-P8-0': 'T8-P8'})
    return eeg_raw

# need to generate timestamps for each file in chb24 since file timestamps do not exist for this
# we will just use the sampling rate to determine file timestamps
# safe assumption here is that each consecutive file is immediately recorded after the last
# the parameter here is the list of the associated patient summary processed dictionary
# (this list includes filenames and metadata)
def generate_timesteps(file_summary_object, patient_name):
    start_seconds = 0
    for i, record in enumerate(file_summary_object):
        filename = list(record.keys())[0]
        eeg_raw = read_mne_data(os.path.join('.', 'Data', patient_name, filename))
        sampling_frequency = int(eeg_raw.info['sfreq'])

        # getting data in the shape (num_channels,samples) where we sample 'sampling_frequency * seconds'
        eeg_raw_data = eeg_raw.get_data()
        num_samples = eeg_raw_data.shape[1]
        # dividing samples by the sampling rate to get the time elapsed in seconds
        seconds_elapsed = int(num_samples / sampling_frequency)
        start_time = start_seconds
        '''
        start_min, mod_sec = divmod(start_seconds, 60)
        start_hour, start_min = divmod(start_min, 60)
        start_time = (start_hour, start_min, mod_sec)
        '''
        # incrementing start time with elapsed time
        start_seconds += seconds_elapsed
        end_time = start_seconds
        '''
        end_min, mod_sec = divmod(start_seconds, 60)
        end_hour, end_min = divmod(end_min, 60)
        end_time = (end_hour, end_min, mod_sec)
        '''
        # updating list
        file_summary_object[i][filename][0] = (start_time, end_time)
    # returning the modified file_summary_object (this should include non-NONE timestamps)
    return file_summary_object


# need to grab data for a patient and split it into ictal and inter-ictal segments
# we can then window and process these segments for classification
# for comparison to other works, inter-ictal segments should be extracted at least 4 hours after or before a seizure
def split_eeg_into_classes(patient_metadata_list):
    # we need to grab a list of absolute times of seizures first
    seizure_absolutes = []
    for file_data in patient_metadata_list:
        time_data = file_data[file_data.keys()[0]]
        start_sec = time_data[0][0]
        for i in range(1,len(time_data)):
            seizure_start_adjusted = time_data[i][0] + start_sec
            seizure_end_adjusted = time_data[i][1] + start_sec
            seizure_absolutes.append((seizure_start_adjusted,seizure_end_adjusted))

    # now that we have seizure absolute timesteps for this patient, we can extract ictal and interictal from each file
    # reminder, we extract all ictal data and only interictal data 4 hours before or after any seizure
    ictal_segments = []
    interictal_segments = []
    for file_data in patient_metadata_list:
        pass


# function that takes an eeg recording file and splits it into windows based on a window size and sampling frequency
# NOTE: window size is in seconds-> number of samples will be taken care of automagically by this function
# seizure times is the list of tuples [(start_time,end_time),(start_time_2,end_time_2),...]
# TODO: implement tagging here
def window_recordings(file_path, seizure_times, window_size=12):
    eeg_raw = read_mne_data(file_path)
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
def stft_recordings(eeg_data, sampling_frequency=256, window=256, overlap=None):
    # applying stft to data
    _, _, frequencies = signal.stft(eeg_data, fs=sampling_frequency, nperseg=256, noverlap=overlap)
    # removing the start and end times to be consistent with the paper
    frequencies = frequencies[:, :, 1:-1]
    # removing DC component (0 Hz), the 57-63Hz and 117-123Hz bands (specific for chb-mit dataset)
    frequencies = np.delete(frequencies, [0, *[i for i in range(57, 64, 1)], *[i for i in range(117, 124, 1)]], axis=1)
    return frequencies


# function that validates whether we have grabbed all records
# this returns a list of records which are provided in the dataset for chb-mit, but we haven't grabbed
# ideally, this list should be empty
def validate_records(record_list_path, summary_dictionary):
    missed_records = []
    # getting all record names in the summary dictionary first
    record_names = {}
    for patient_name in summary_dictionary:
        for record in summary_dictionary[patient_name]:
            edf_file = list(record.keys())[0]
            record_names[edf_file] = 1
    file = open(record_list_path, 'r')
    edf_file_list = file.readlines()
    for filename in edf_file_list:
        filename = os.path.basename(filename.strip())
        if filename not in record_names:
            missed_records.append(filename)
    return missed_records


# the only missing records is in patient 24 due to annotators not putting all data in the summary file
# so, we need to add these records to the processed summary dictionary
# the argument is the specific record list for patient 24
def grab_missing_records(record_list):
    new_record_list = []
    missing_records = ['chb24_02.edf', 'chb24_05.edf', 'chb24_08.edf', 'chb24_10.edf',
                       'chb24_12.edf', 'chb24_16.edf', 'chb24_18.edf', 'chb24_19.edf', 'chb24_20.edf', 'chb24_22.edf']
    missing_records = list(map(lambda x: {x: [((None, None, None), (None, None, None))]}, missing_records))
    # using "finger" method to create new list in sorted order
    idx1, idx2 = 0, 0
    while idx1 < len(missing_records) and idx2 < len(record_list):
        # comparing end of filenames to see which record goes first
        # i.e chb24_01 comes before chb24_02
        name1 = list(missing_records[idx1].keys())[0]
        name2 = list(record_list[idx2].keys())[0]
        if int(name1[6:8]) < int(name2[6:8]):
            # then idx1 is incremented
            new_record_list.append(missing_records[idx1])
            idx1 += 1
        else:
            new_record_list.append(record_list[idx2])
            idx2 += 1
    while idx1 < len(missing_records):
        new_record_list.append(missing_records[idx1])
        idx1 += 1
    while idx2 < len(record_list):
        new_record_list.append(record_list[idx2])
        idx2 += 1
    return new_record_list


'''
    Below is some basic logic I wrote while testing stuff
'''

patient_metadata = preprocess_data()
test_list = patient_metadata['chb01']
print(patient_metadata)
'''
files = get_eegs()[0]
window_test = window_recordings(files,None)[0]
test_fft = stft_recordings(window_test)
print(test_fft.shape)
'''

'''
# read in an edf file
test = mne.io.read_raw_edf('./Data/chb14/chb14_01.edf',exclude=['-'])
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
test_data = test_data[:,:256*12]
print(test_data.shape)

# short time fast fourier transform
#test_fft = mne.time_frequency.stft(test_data,wsize=256)
f,t,test_fft = signal.stft(test_data,256)
print(f)
print(t)
print(test_fft.shape)
# (22,129,7200)
# 22 channels, 129 frequencies (0 to 128), and 7200 timesteps by shifting half a window each time
print(test_fft.shape)

# removing DC component (0 Hz), the 57-63Hz and 117-123Hz bands
test_fft = np.delete(test_fft,[0,*[i for i in range(57,64,1)],*[i for i in range(117,124,1)]],axis=1)

print(test_fft.shape)
'''
