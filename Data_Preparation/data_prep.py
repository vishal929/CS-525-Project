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
def preprocess_metadata():
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
            final_mapping[os.path.dirname(summary_file)] = record_list
            # final_mapping[os.path.basename(summary_file)[:5]] = record_list
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

    return final_mapping

# function that returns true if a directory is over the given size threshold
# this will help in pruning our huge data with 12s windows
# if threshold is none, then we do no pruning
def dir_over_limit(folder_path,threshold=None,window_size=12):
    if threshold is None:
        return False
    else:
        tot_size = 0
        for entry in os.scandir(folder_path):
            # checking if this file corresponds to our window size (if not, then we dont count it in the threshold)
            if entry.name[0:2] == str(window_size):
                tot_size += os.path.getsize(entry)
        if tot_size > threshold:
            return True
        else:
            return False

# we process leave out_one_data here
# basically, for each patient, we save all interictal and ictal samples
# however, for ictal samples, we separate files based on seizure instance
# i.e one seizure will be in a single file, one seizure in another and so on
def process_leave_out_one_data(window_size=1,size_threshold=None):
    metadata = preprocess_metadata()
    for patient in metadata:
        # patient 12 and 24 should be omitted due to no valid interictal data
        if 'chb12' in patient or 'chb24' in patient:
            continue
        # window size 12 comparisons exclude patient 16
        if window_size == 12 and 'chb16' in patient:
            continue
        # getting patient id, like 'chb01' or 'chb02'
        patient_id = os.path.basename(patient)
        patient_files = metadata[patient]
        ictal, interictal = split_eeg_into_classes(patient_files, patient)
        ictal_train, ictal_val = [], []
        interictal_train, interictal_val = [], []

        # need to create directory if it doesnt already exist
        if not os.path.isdir(os.path.join('.', 'Processed_Data', patient_id)):
            os.mkdir(os.path.join('.', 'Processed_Data', patient_id))

        # after 1000 examples, we want to create a separate batched file (only applicable for interictal)
        if window_size == 1:
            save_threshold = 12000
        else:
            save_threshold = 1000

        train_save_count = 0
        while len(ictal) > 0:
            segment = ictal.pop()
            # we want to window this segment, run short fourier transform and send to train/val
            # the latter 25% are sent to validation
            # WHAT IS IMPORTANT HERE IS THAT SEIZURE DATA IS INDEXED! SO WE CAN DO CROSS VALIDATION ON SEIZURE LEVEL FOR EACH PATIENT
            windows = window_recordings(segment, window_size=window_size)
            if windows.shape[0] == 0:
                # empty array (window size is larger than the segment)
                continue
            train, val = np.split(windows, [int(len(windows) * 0.75)])
            train_save_count += 1
            # we save this ictal segment
            np.save(os.path.join('.', 'Processed_Data', patient_id,
                                     str(window_size) + '-' + str(train_save_count) + '-ictal_train.npy'),
                        train)
            np.save(os.path.join('.', 'Processed_Data', patient_id,
                                 str(window_size) + '-' + str(train_save_count) + '-ictal_val.npy'),
                    val)



        # most of our values have around 9 digits of precision and exponent around -05 to -08, so float32 is all we need
        # saving ictal data to disk (we are saving as float32, float64 is going to be worse in our case

        num_train_processed = 0
        num_val_processed = 0
        train_save_count = 0
        val_save_count = 0
        while len(interictal) > 0:
            segment = interictal.pop()
            # the latter 25% are sent to validation
            # we want to window this segment, run short fourier transform and send to train/val
            windows = window_recordings(segment, window_size=window_size)
            if windows.shape[0] == 0:
                # empty array (window size is larger than the segment)
                continue
            train, val = np.split(windows, [int(len(windows) * 0.75)])

            # adding to data pool
            interictal_train.append(train)
            interictal_val.append(val)

            num_train_processed += train.shape[0]
            num_val_processed += val.shape[0]

            if num_train_processed > save_threshold or len(interictal) == 0:
                # we should save this array
                interictal_train = np.concatenate(interictal_train, axis=0, dtype=np.float32)
                # 22 channels
                # assert interictal_train.shape[1] == 22
                # 114 frequencies in frequency domain
                # assert interictal_train.shape[2] == 114
                train_save_count += 1
                np.save(os.path.join('.', 'Processed_Data', patient_id,
                                     str(window_size) + '-' + str(train_save_count) + '-interictal_train.npy'),
                        interictal_train)
                # checking if we are over the limit
                if dir_over_limit(os.path.join('.', 'Processed_Data', patient_id), threshold=size_threshold,
                                  window_size=window_size):
                    break
                # resetting
                interictal_train = []
                num_train_processed = 0

            if num_val_processed > save_threshold or len(interictal) == 0:
                # we should save this array
                interictal_val = np.concatenate(interictal_val, axis=0, dtype=np.float32)
                # 22 channels
                # assert interictal_val.shape[1] == 22
                # 114 frequencies in frequency domain
                # assert interictal_val.shape[2] == 114
                val_save_count += 1
                np.save(os.path.join('.', 'Processed_Data', patient_id, str(window_size) + '-' + str(val_save_count) \
                                     + '-interictal_val.npy'),
                        interictal_val)
                # checking if we are over the limit
                if dir_over_limit(os.path.join('.', 'Processed_Data', patient_id), threshold=size_threshold,
                                  window_size=window_size):
                    break
                # resetting
                interictal_val = []
                num_val_processed = 0


# function that processes,tags, and splits data eeg data for seizure classification based on processed metadata
# window size is the window to generate sequences from ictal and interictal segments
# threshold is the max size of data we want for each patient (we set this so that we do not get huge data for compute)
# if threshold is set to None, there is no threshold (no limit)
def process_data(window_size=1,size_threshold=None):
    metadata = preprocess_metadata()
    for patient in metadata:
        # patient 12 and 24 should be omitted due to no valid interictal data
        if 'chb12' in patient or 'chb24' in patient:
            continue
        # window size 12 comparisons exclude patient 16
        if window_size == 12 and 'chb16' in patient:
            continue
        # getting patient id, like 'chb01' or 'chb02'
        patient_id = os.path.basename(patient)
        patient_files = metadata[patient]
        ictal, interictal = split_eeg_into_classes(patient_files, patient)
        ictal_train, ictal_val = [], []
        interictal_train, interictal_val = [], []

        # need to create directory if it doesnt already exist
        if not os.path.isdir(os.path.join('.', 'Processed_Data', patient_id)):
            os.mkdir(os.path.join('.', 'Processed_Data', patient_id))

        # after 1000 examples, we want to create a separate batched file
        if window_size==1:
            save_threshold = 12000
        else:
            save_threshold = 1000

        num_train_processed = 0
        num_val_processed = 0
        train_save_count = 0
        val_save_count = 0
        while len(ictal) > 0:
            segment = ictal.pop()
            # we want to window this segment, run short fourier transform and send to train/val
            # the latter 25% are sent to validation
            windows = window_recordings(segment, window_size=window_size)
            if windows.shape[0] == 0:
                # empty array (window size is larger than the segment)
                continue
            # 75% sent to train, 25% sent to validation
            train, val = np.split(windows, [int(len(windows) * 0.75)])
            #train, val = stft_recordings(train), stft_recordings(val)

            # adding to data pool
            ictal_train.append(train)
            ictal_val.append(val)

            num_train_processed += train.shape[0]
            num_val_processed += val.shape[0]

            if num_train_processed > save_threshold or len(ictal) == 0:
                # we should save this array
                ictal_train = np.concatenate(ictal_train, axis=0, dtype=np.float32)
                # 22 channels
                #assert ictal_train.shape[1] == 22
                # 114 frequencies in frequency domain
                #assert ictal_train.shape[2] == 114
                train_save_count += 1
                np.save(os.path.join('.', 'Processed_Data', patient_id, str(window_size) + '-'+str(train_save_count)+'-ictal_train.npy'),
                        ictal_train)
                # checking if we are over the limit
                if dir_over_limit(os.path.join('.', 'Processed_Data', patient_id),threshold=size_threshold,window_size=window_size):
                    break
                # resetting
                ictal_train = []
                num_train_processed=0

            if num_val_processed > save_threshold or len(ictal)==0:
                # we should save this array
                ictal_val = np.concatenate(ictal_val, axis=0, dtype=np.float32)
                # 22 channels
                #assert ictal_val.shape[1] == 22
                # 114 frequencies in frequency domain
                #assert ictal_val.shape[2] == 114
                val_save_count += 1
                np.save(os.path.join('.', 'Processed_Data', patient_id, str(window_size) + '-' + str(val_save_count) \
                                     + '-ictal_val.npy'),
                        ictal_val)
                # checking if we are over the limit
                if dir_over_limit(os.path.join('.', 'Processed_Data', patient_id), threshold=size_threshold,window_size=window_size):
                    break
                # resetting
                ictal_val = []
                num_val_processed=0

        # most of our values have around 9 digits of precision and exponent around -05 to -08, so float32 is all we need
        # saving ictal data to disk (we are saving as float32, float64 is going to be worse in our case

        num_train_processed = 0
        num_val_processed = 0
        train_save_count = 0
        val_save_count = 0
        while len(interictal)>0:
            segment = interictal.pop()
            # the latter 25% are sent to validation
            # we want to window this segment, run short fourier transform and send to train/val
            windows = window_recordings(segment, window_size=window_size)
            if windows.shape[0] == 0:
                # empty array (window size is larger than the segment)
                continue
            train, val = np.split(windows, [int(len(windows) * 0.75)])

            #train, val = stft_recordings(train), stft_recordings(val)

            # adding to data pool
            interictal_train.append(train)
            interictal_val.append(val)

            num_train_processed += train.shape[0]
            num_val_processed += val.shape[0]

            if num_train_processed > save_threshold or len(interictal) == 0:
                # we should save this array
                interictal_train = np.concatenate(interictal_train, axis=0, dtype=np.float32)
                # 22 channels
                #assert interictal_train.shape[1] == 22
                # 114 frequencies in frequency domain
                #assert interictal_train.shape[2] == 114
                train_save_count += 1
                np.save(os.path.join('.', 'Processed_Data', patient_id,
                                     str(window_size) + '-' + str(train_save_count) + '-interictal_train.npy'),
                        interictal_train)
                # checking if we are over the limit
                if dir_over_limit(os.path.join('.', 'Processed_Data', patient_id), threshold=size_threshold,window_size=window_size):
                    break
                # resetting
                interictal_train = []
                num_train_processed=0

            if num_val_processed > save_threshold or len(interictal) == 0:
                # we should save this array
                interictal_val = np.concatenate(interictal_val, axis=0, dtype=np.float32)
                # 22 channels
                #assert interictal_val.shape[1] == 22
                # 114 frequencies in frequency domain
                #assert interictal_val.shape[2] == 114
                val_save_count += 1
                np.save(os.path.join('.', 'Processed_Data', patient_id, str(window_size) + '-' + str(val_save_count) \
                                     + '-interictal_val.npy'),
                        interictal_val)
                # checking if we are over the limit
                if dir_over_limit(os.path.join('.', 'Processed_Data', patient_id), threshold=size_threshold,window_size=window_size):
                    break
                # resetting
                interictal_val = []
                num_val_processed=0


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

                start_object = datetime.datetime.strptime(str(s_hour) + ':' + str(s_min) + ':' + str(s_sec), '%H:%M:%S')
                end_object = datetime.datetime.strptime(str(e_hour) + ':' + str(e_min) + ':' + str(e_sec), '%H:%M:%S')
                # need to account for the day to avoid negative differences
                while last_end_time is not None and start_object < last_end_time:
                    # need to increment the day of this object by 1 i.e we are going from something like 23 hr to 2 hr
                    start_object += datetime.timedelta(days=1)

                while end_object < start_object:
                    # need to account for wraparound, we can just increment the day of end_object by 1
                    end_object += datetime.timedelta(days=1)

                # getting timedelta in seconds
                seconds_elapsed = int((end_object - start_object).total_seconds())
                if last_end_time is None:
                    start_seconds = 0
                    end_seconds = seconds_elapsed
                else:
                    seconds_elapsed_between_files = int((start_object - last_end_time).total_seconds())
                    start_seconds = last_absolute_end_time + seconds_elapsed_between_files
                    end_seconds = start_seconds + seconds_elapsed
                last_end_time = end_object
                last_absolute_end_time = end_seconds

            if len(seizure_mapping) != 0:
                total_files.append(seizure_mapping)
                seizure_mapping = {}
            seizure_mapping[curr_filename] = [(start_seconds, end_seconds)]
        if 'seizure' in line and 'start' in line:
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
    # we need to standardize the channels we want to grab

    channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4',
                'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'P7-T7', 'T7-FT9', 'FT9-FT10',
                'FT10-T8']

    # reading the eeg file and excluding dummy '-' channels and 'ECG' Channels (not all patients have ECG recordings)
    eeg_raw = mne.io.read_raw_edf(filename, include=channels)

    # removing the redundant channel
    if 'T8-P8-1' in eeg_raw.info['ch_names']:
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
# par_path is the path of the patient directory associated with the metadata list (i.e './Data/chb01/)
def split_eeg_into_classes(patient_metadata_list, par_path):
    # we need to grab a list of absolute times of seizures first
    seizure_absolutes = []
    for file_data in patient_metadata_list:
        time_data = file_data[list(file_data.keys())[0]]
        start_sec = time_data[0][0]
        for i in range(1, len(time_data)):
            seizure_start_adjusted = time_data[i][0] + start_sec
            seizure_end_adjusted = time_data[i][1] + start_sec
            seizure_absolutes.append((seizure_start_adjusted, seizure_end_adjusted))
    # print(len(seizure_absolutes))
    # creating a dictionary of invalid timesteps for quick lookup
    # 4 hours is 14400 seconds
    invalid_times = set([])
    for seizure in seizure_absolutes:
        seizure_start = seizure[0]
        seizure_end = seizure[1]
        # invalid range is not inclusive
        invalid = np.arange(seizure_start - 14400 + 1, seizure_end + 14400 - 1)
        invalid_times.update(set(invalid))
    # print(len(invalid_times))

    # print('invalid times set created!')
    # now that we have seizure absolute timesteps for this patient, we can extract ictal and interictal from each file
    # reminder, we extract all ictal data and only interictal data 4 hours before or after any seizure
    ictal_segments = []
    interictal_segments = []
    for file_data in patient_metadata_list:
        file_name = list(file_data.keys())[0]
        file_path = os.path.join(par_path, file_name)
        time_data = file_data[file_name]
        start_sec = time_data[0][0]
        end_sec = time_data[0][1]
        # 4 hours is 14400 seconds
        # getting any seizure data first if it exists
        for i in range(1, len(time_data)):
            # extracting ictal sections
            ictal = extract_section(file_path, time_data[i])
            # ictal may be none if the edf file has different channels than the ones we are looking for (invalid data)
            if ictal is not None:
                ictal_segments.append(ictal)

        curr_range = (None, None)
        total_range = np.arange(start_sec, end_sec + 1)
        for idx, i in enumerate(total_range):
            # creating a range of values that are valid
            if i not in invalid_times:
                if curr_range[0] is None:
                    curr_range = (idx, None)
                else:
                    curr_range = (curr_range[0], idx)
            else:
                # then we hit an invalid timestep
                if curr_range[0] is not None and curr_range[1] is not None:
                    interictal = extract_section(file_path, curr_range)
                    # interictal may be none if the file has invalid channels (not like the rest)
                    if interictal is not None:
                        interictal_segments.append(interictal)
                curr_range = (None, None)
        if curr_range[0] is not None and curr_range[1] is not None:
            # interictal may be none if the edf file has different channels than the ones we are looking for (invalid data)
            interictal = extract_section(file_path, curr_range)
            if interictal is not None:
                interictal_segments.append(interictal)
    return ictal_segments, interictal_segments


# function that extracts numpy array of an eeg section given a .edf filename and a tuple of (start,end)
def extract_section(file_path, start_end):
    raw_eeg = read_mne_data(file_path)
    sampling_rate = int(raw_eeg.info['sfreq'])
    try:
        data = raw_eeg.get_data()
    except ValueError:
        # we may get value error with invalid channel types (edf recording is not like the other recordings)
        return None
    # data is in shape (num_channels,samples)
    final_data = data[:, sampling_rate * start_end[0]:sampling_rate * start_end[1]]
    # channel check for invalid data
    if final_data.shape[0] != 22:
        return None
    return final_data


# function that takes an eeg numpy array and splits it into windows based on the window size
# NOTE: window size is in seconds-> number of samples will be taken care of automagically by this function
def window_recordings(eeg_raw_data, sampling_frequency=256, window_size=12):
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

    # returning an array of windows
    return np.array(new_data)


# function that takes an eeg window, and applies the short term fourier transform with window and overlap parameters
# we filter out noise frequency and DC frequency specific to the CHB-MIT scalp EEG dataset
# this function can take a batch of eeg windows as long as the batch is the first dimension
'''
    DO NOT USE THIS FUNCTION! INSTEAD WE SAVE THE RAW WINDOWED SAMPLES TO DISK
    THEN, WHEN LOADING THE SAMPLES INTO TENSORFLOW, WE USE TF.STFT (much more efficient in our testing)
'''
def stft_recordings(eeg_window, sampling_frequency=256, window=256, overlap=None):
    # applying stft to data
    _, _, frequencies = signal.stft(eeg_window, fs=sampling_frequency, nperseg=window, noverlap=overlap)
    # removing the start and end times from the time dimension (last dimension) to be consistent with the paper
    frequencies = np.delete(frequencies, [0, frequencies.shape[len(frequencies.shape) - 1] - 1], axis=-1)
    # frequencies = frequencies[:, :, 1:-1]

    # removing DC component (0 Hz), the 57-63Hz and 117-123Hz bands (specific for chb-mit dataset)
    frequencies = np.delete(frequencies, [0, *[i for i in range(57, 64, 1)], *[i for i in range(117, 124, 1)]], axis=-2)

    # stft returns complex numbers, so we use np.abs to obtain their magnitude
    frequencies = np.abs(frequencies)

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
            # sanity check that our times make sense as well (are positive)
            for time_tuple in record[edf_file]:
                assert (time_tuple[0] >= 0 and time_tuple[1] >= 0)
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

process_leave_out_one_data(window_size=1,size_threshold=None)
