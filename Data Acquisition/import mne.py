import mne
import pandas as pd
import numpy as np

User_frame = pd.read_csv("D:\\Users\\EEG\\Desktop\\projetEEG\\droite\\EEG.csv", sep=',', index_col=0) 
data = User_frame.transpose().to_numpy() 

# assigning the channel type when initializing the Info object
ch_names = ['sequence','battery','flags','EEG-ch1','EEG-ch2','EEG-ch3','EEG-ch4','EEG-ch5'
    ,'EEG-ch6','EEG-ch7','EEG-ch8','EEG-ch9','EEG-ch10']

ch_types = ['misc','misc','misc', 'eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg']

sampling_freq = 256  # in Hertz

info = mne.create_info(ch_names= ch_names, ch_types= ch_types, sfreq= sampling_freq)

User_raw = mne.io.RawArray(data, info)

print('gggg ',User_raw.info['ch_names'])