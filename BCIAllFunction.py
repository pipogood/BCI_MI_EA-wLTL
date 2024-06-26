import mne
import numpy as np
from mne.datasets import eegbci
import matplotlib.pyplot as plt
from os import listdir
from mne.channels import make_standard_montage
from scipy import signal

class BCIFuntions:
    def __init__(self):
        pass
    
    def GetRawGDF(self, name="pipo", con="offline"):
        if con == "offline":
            condition = "Offline_Experiment"
        elif con == "online":
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

    def butter_bandpass_filter(self,data,lowcut,highcut,fs,order):
        b,a = self.butter_bandpass(lowcut,highcut,fs,order)
        y = signal.filtfilt(b,a,data,axis=2)
        return y