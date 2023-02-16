import numpy as np
import matplotlib.pyplot as plt
import mne

# chb-mit includes 23 channels sampled at 256hz


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


