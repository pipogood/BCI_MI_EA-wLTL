import mne
import numpy as np
from mne.datasets import eegbci
import matplotlib.pyplot as plt
from os import listdir
from mne.channels import make_standard_montage
from scipy import signal
from scipy.linalg import sqrtm, inv 

class BCIFuntions:
    def __init__(self, numclass, frequency):
        self.numclass = numclass
        self.fs = frequency
    
    def GetRawGDF(self, name="pipo", condition="offline"):
        if condition == "offline":
            condition = "Offline_Experiment"
        elif condition == "online":
            condition = "Online_Experiment"

        path = "C:\\git\Senior_Thesis\\DataSet\\"+condition+"\\"+ name +"\\iir_car\\"
        list_dir = listdir(path)
        raw_each = [0] * len(list_dir)
        for i in range(len(list_dir)):
            raw_each[i] = mne.io.read_raw_gdf(path+list_dir[i],preload = False)
            
        raw_gdf = mne.concatenate_raws(raw_each)

        eegbci.standardize(raw_gdf)  # set channel names
        montage = make_standard_montage("standard_1005")
        raw_gdf.set_montage(montage)

        print(f"Successful to create Data of {name}")

        return raw_gdf
    
    def GetEpoch(self, raw_gdf, tmin=-2.0, tmax=6.0, baseline = (-0.5,0.0), event_repeat = True):
        events, event_dict = mne.events_from_annotations(raw_gdf)
        event_dict =  {'OVTK_GDF_Left': 8,
        'OVTK_GDF_Right': 9,
        'OVTK_GDF_Tongue': 10,
        'OVTK_GDF_Up': 11}

        if event_repeat:
            event_dict =  {'OVTK_GDF_Left': 1,
            'OVTK_GDF_Right': 2,
            'OVTK_GDF_Tongue': 3,
            'OVTK_GDF_Up': 4}
            events_1 = np.delete(events, [0], axis= 0)
            arr2= np.arange(len(events_1))
            events = events_1[(arr2 % 5 == 0)]

        Epochs = mne.Epochs(raw_gdf, events, 
            tmin= tmin,  
            tmax= tmax,    
            event_id=event_dict,
            preload = True,
            event_repeated='drop',
            baseline=baseline
            )

        return Epochs

    def butter_bandpass(self,lowcut,highcut,fs,order):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = signal.butter(order,[low,high],'bandpass')
        return b,a

    def butter_bandpass_filter(self,data,lowcut = 6,highcut = 30, order = 4):
        b,a = self.butter_bandpass(lowcut,highcut,self.fs,order)
        y = signal.filtfilt(b,a,data,axis=2)
        return y
    

    def GetRawSet_ComputeEA(self, filtered_epochs, label):
        '''
        Note: EEG labels 
        class 1 is Left hand imagery
        class 2 is Right hand imagery
        class 3 is Feet imagery
        class 4 is Non-imagery

        For motor imagery, we will crop data at 0-4 second after direction cue represent and filter 6-32 Hz

        This function will return raw_EEG and EA_EEG
        '''

        Raw_data = {}
        Raw_data["EEG_left"] = filtered_epochs[np.where(label == 1)]
        Raw_data["EEG_right"] = filtered_epochs[np.where(label == 2)]
        Raw_data["EEG_non"] = filtered_epochs[np.where(label == 3)]
        Raw_data["EEG_feet"] = filtered_epochs[np.where(label == 4)]

        # Calculate reference matrix
        RefEA = 0

        # Iterate over all trials, compute reference EA
        for trial in filtered_epochs:
            cov = np.cov(trial, rowvar=True)
            RefEA += cov

        # Average over all trials
        RefEA = RefEA/filtered_epochs.shape[0]

        # Add to data
        EA_data = {}
        EA_data['RefEA'] = RefEA 
        EA_left = []
        EA_right = []
        EA_feet = []
        EA_non = []

        # Compute R^(-0.5)
        R_inv = sqrtm(inv(RefEA))

        for left, right, feet, non in zip(Raw_data["EEG_left"] , Raw_data["EEG_right"] , Raw_data["EEG_feet"] , Raw_data["EEG_non"] ):
            EA_left.append(R_inv@left)
            EA_right.append(R_inv@right)
            EA_feet.append(R_inv@feet)
            EA_non.append(R_inv@non)

        # Store as left_EA and right_EA
        EA_data['EEG_left_EA'] = np.array(EA_left)
        EA_data['EEG_right_EA'] = np.array(EA_right)
        EA_data['EEG_feet_EA'] = np.array(EA_feet)
        EA_data['EEG_non_EA'] = np.array(EA_feet)

        return Raw_data, EA_data

    def plot_rawEA(self, raw_data, EA_data, ch_name, no_trial = 0):
        '''
        This function will plot time series data compare between EA and non-EA 
        Require raw_data, EA_data in a list of 4 classes and selected trial to plot
        '''

        left = raw_data['EEG_left'][no_trial]
        right = raw_data['EEG_right'][no_trial]
        feet = raw_data['EEG_feet'][no_trial]
        noim = raw_data['EEG_non'][no_trial]

        left_alg = EA_data['EEG_left_EA'][no_trial]
        right_alg = EA_data['EEG_right_EA'][no_trial]
        feet_alg = EA_data['EEG_feet_EA'][no_trial]
        non_alg = EA_data['EEG_non_EA'][no_trial]

        # Time
        time = np.arange(raw_data['EEG_left'].shape[2])/self.fs #total time

        # Number of electrodes
        n_el = raw_data['EEG_left'].shape[1]

        # Plot each raw EEG and aligned raw EEG
        fig, axes = plt.subplots(n_el, self.numclass, figsize=(20, 20), sharex=True)

        # Title to each column
        axes[0, 0].set_title('Left')
        axes[0, 1].set_title('Right')
        axes[0, 2].set_title('Feet')
        axes[0, 3].set_title('Non-imagine')

        # Plot each eeg raw and aligned 
        for i, ax in enumerate(axes):
            ax[0].plot(time, left[i], color='k', label= str(ch_name[i]))
            ax[0].plot(time, left_alg[i], color='r', label= str(ch_name[i]))
            ax[0].legend(loc=1)
            
            ax[1].plot(time, right[i], color='k', label= str(ch_name[i]))
            ax[1].plot(time, right_alg[i], color='r', label= str(ch_name[i]))
            ax[1].legend(loc=1)

            ax[2].plot(time, feet[i], color='k', label= str(ch_name[i]))
            ax[2].plot(time, feet_alg[i], color='r', label= str(ch_name[i]))
            ax[2].legend(loc=1)
            
            ax[3].plot(time, noim[i], color='k', label= str(ch_name[i]))
            ax[3].plot(time, non_alg[i], color='r', label= str(ch_name[i]))
            ax[3].legend(loc=1)

        plt.show()




