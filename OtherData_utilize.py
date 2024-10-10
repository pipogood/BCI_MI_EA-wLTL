import mne
import numpy as np
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from scipy import signal
from sklearn.metrics import classification_report,confusion_matrix
from os import listdir
import random
import pickle

class Physionet:
    def __init__(self, selectclass, desired_fz, ch_pick):
        self.allclass = selectclass
        self.fs = desired_fz
        self.picks = ch_pick
        random.seed(42)
        self.class_name = np.array(["Left", "Right", "Non", "Feet"])

    def butter_bandpass(self,lowcut,highcut,fs,order):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = signal.butter(order,[low,high],'bandpass')
        return b,a

    def butter_bandpass_filter(self, data,lowcut = 6, highcut = 30, order = 4):
        b,a = self.butter_bandpass(lowcut,highcut,self.fs,order)
        y = signal.filtfilt(b,a,data,axis=2)
        return y
    
    def GetConfusionMatrix(self, models, X_train, X_test, y_train, y_test):
        y_pred = models.predict(X_train)
        print("Classification TRAIN DATA \n=======================")
        print(classification_report(y_true= y_train, y_pred=y_pred, target_names=self.class_name[np.unique(y_train)]))
        print("Confusion matrix \n=======================")
        print(confusion_matrix(y_true= y_train, y_pred=y_pred))

        y_pred = models.predict(X_test)
        print("Classification TEST DATA \n=======================")
        print(classification_report(y_true=y_test, y_pred=y_pred, target_names=self.class_name[np.unique(y_test)]))
        print("Confusion matrix \n=======================")
        print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    def GetRaw(self, num_subject = 30, start_subject = 0):

        raw_RorL1sub = [0]*num_subject  # RorL is Right or Left fist movement/imagery
        raw_Both1sub = [0]*num_subject  # Both is both feet and both fits movement/imagery

        start_subject = 0

        RAW_data_RorL = {}
        RAW_data_Both = {}

        for j in range(1+start_subject,num_subject+1+start_subject):

            print("processing subject number: ", j)

            if j < 10:
                subject  = '00' + str(j)
            elif j < 100:
                subject = '0' + str(j)
            else:
                subject = str(j)

            raw_RorL1 = mne.io.read_raw_edf("D:\physionet_dataset\S" + str(subject) +"\S" + str(subject) +"R04.edf",preload = True, verbose=False)
            raw_RorL2 = mne.io.read_raw_edf("D:\physionet_dataset\S" + str(subject) +"\S" + str(subject) +"R08.edf",preload = True, verbose=False)
            raw_RorL3 = mne.io.read_raw_edf("D:\physionet_dataset\S" + str(subject) +"\S" + str(subject) +"R12.edf",preload = True, verbose=False)

            raw_RorL1sub[j-1+start_subject] = mne.concatenate_raws([raw_RorL1.pick(self.picks), raw_RorL2.pick(self.picks), raw_RorL3.pick(self.picks)])
            raw_RorL1sub[j-1+start_subject] = raw_RorL1sub[j-1+start_subject].resample(128) 

            eegbci.standardize(raw_RorL1sub[j-1+start_subject])  # set channel names
            montage = make_standard_montage("standard_1005")    
            raw_RorL1sub[j-1+start_subject].set_montage(montage)

            RAW_data_RorL["P" + str(subject)] = {"Raw_data": raw_RorL1sub[j-1+start_subject].copy()}

            raw_Both1 = mne.io.read_raw_edf("D:\physionet_dataset\S" + str(subject) +"\S" + str(subject) +"R06.edf",preload = True, verbose=False)
            raw_Both2 = mne.io.read_raw_edf("D:\physionet_dataset\S" + str(subject) +"\S" + str(subject) +"R10.edf",preload = True, verbose=False)
            raw_Both3 = mne.io.read_raw_edf("D:\physionet_dataset\S" + str(subject) +"\S" + str(subject) +"R14.edf",preload = True, verbose=False)

            raw_Both1sub[j-1+start_subject] = mne.concatenate_raws([raw_Both1.pick(self.picks), raw_Both2.pick(self.picks), raw_Both3.pick(self.picks)])
            raw_Both1sub[j-1+start_subject] = raw_Both1sub[j-1+start_subject].resample(128)

            eegbci.standardize(raw_Both1sub[j-1+start_subject])  # set channel names
            montage = make_standard_montage("standard_1005")    
            raw_Both1sub[j-1+start_subject].set_montage(montage)

            RAW_data_Both["P" + str(subject)] = {"Raw_data": raw_Both1sub[j-1+start_subject].copy()}

        return RAW_data_RorL, RAW_data_Both
    
    def Get_epoch(self, RAW_data_RorL, RAW_data_Both, tmin=-2.0, tmax=4.0, crop=(0,2),baseline = (-0.5,0.0), band_pass = (6,32) ,trial_removal_th = 100):
        EEG_epoch = {}
        for key_subs in RAW_data_RorL:
            event_id_mapping_RorL = {
                old_event_id: new_event_id
                for old_event_id, new_event_id in zip([1, 2, 3], [2, 0, 1]) 
            }

            events_RorL, event_id1 = mne.events_from_annotations(RAW_data_RorL[key_subs]['Raw_data'], verbose=False)
            events_RorL[:, 2] = [event_id_mapping_RorL.get(event_id1, event_id1) for event_id1 in events_RorL[:, 2]]

            event_id1 = {'Rest': 2, 'Left': 0, 'Right': 1 } 

            RorL_epochs = mne.Epochs(RAW_data_RorL[key_subs]['Raw_data'], events_RorL, 
                tmin= tmin,    
                tmax= tmax,   
                event_id= event_id1,
                preload = True,
                event_repeated='drop',
                baseline=baseline,
                verbose = False
            )

            ########################################################################################################

            event_id_mapping_Both = {
                old_event_id: new_event_id
                for old_event_id, new_event_id in zip([1, 2, 3], [2, 4, 3])
            }

            events_Both, event_id2 = mne.events_from_annotations(RAW_data_Both[key_subs]['Raw_data'], verbose=False)
            events_Both[:, 2] = [event_id_mapping_Both.get(event_id2, event_id2) for event_id2 in events_Both[:, 2]]

            event_id2 = {'Rest': 2,'both_feet': 3}  # We Don't use both fits

            Both_epochs = mne.Epochs(RAW_data_Both[key_subs]['Raw_data'], events_Both, 
                tmin= tmin,     # init timestamp of epoch (0 means trigger timestamp same as event start)
                tmax= tmax,    # final timestamp (10 means set epoch duration 10 second)
                event_id= event_id2,
                preload = True,
                event_repeated='drop',
                baseline=baseline,
                verbose = False
            )

            combine_epoch = mne.concatenate_epochs([RorL_epochs, Both_epochs])
            EEG_epoch[key_subs] =  {"Raw_Epoch": combine_epoch.copy().crop(tmin= crop[0], tmax= crop[1])}
            train_data = EEG_epoch[key_subs]['Raw_Epoch'].copy().get_data() * 10e5
            labels = EEG_epoch[key_subs]["Raw_Epoch"].copy().events[:,-1]

            #Reject trial that value over threshold 
            outlier_trial = []
            for ii in range(0,train_data.shape[0]):
                if train_data[ii].max() > trial_removal_th or train_data[ii].min() < -trial_removal_th:
                    outlier_trial.append(ii)
                    print(key_subs,train_data[ii].min(), ii)
                    print(key_subs,train_data[ii].max(), ii)

            EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(train_data, outlier_trial, axis = 0)
            EEG_epoch[key_subs]['label'] = np.delete(labels, outlier_trial)

            filtered_data = self.butter_bandpass_filter(EEG_epoch[key_subs]['Raw_Epoch'], lowcut= band_pass[0], highcut= band_pass[1])

            #Randomly select rest class
            random_delete = random.sample(list(np.where(EEG_epoch[key_subs]['label']== 2)[0]), 60)

            EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(filtered_data, random_delete, axis = 0)
            EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], random_delete)

            #Classes selection
            if "Left" not in self.allclass:
                EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(EEG_epoch[key_subs]['Raw_Epoch'], np.where(EEG_epoch[key_subs]['label']== 0), axis = 0)
                EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], np.where(EEG_epoch[key_subs]['label']== 0))

            if "Right" not in self.allclass:
                EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(EEG_epoch[key_subs]['Raw_Epoch'], np.where(EEG_epoch[key_subs]['label']== 1), axis = 0)
                EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], np.where(EEG_epoch[key_subs]['label']== 1))

            if "Non" not in self.allclass:
                EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(EEG_epoch[key_subs]['Raw_Epoch'], np.where(EEG_epoch[key_subs]['label']== 2), axis = 0)
                EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], np.where(EEG_epoch[key_subs]['label']== 2))

            if "Feet" not in self.allclass:
                EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(EEG_epoch[key_subs]['Raw_Epoch'], np.where(EEG_epoch[key_subs]['label']== 3), axis = 0)
                EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], np.where(EEG_epoch[key_subs]['label']== 3))
            
        return EEG_epoch
    

class BCIcompet2a:
    def __init__(self, selectclass, desired_fz, ch_pick):
        self.allclass = selectclass
        self.fs = desired_fz
        self.picks = ch_pick
        random.seed(42)
        self.class_name = np.array(["Left", "Right", "Non", "Feet"])

    def butter_bandpass(self,lowcut,highcut,fs,order):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = signal.butter(order,[low,high],'bandpass')
        return b,a

    def butter_bandpass_filter(self, data,lowcut = 6, highcut = 30, order = 4):
        b,a = self.butter_bandpass(lowcut,highcut,self.fs,order)
        y = signal.filtfilt(b,a,data,axis=2)
        return y

    def GetRaw(self, target_subjects="all", preload = True):
        '''
        if preload == True: load data which already resample to 128 
        '''
        if preload:
            with open('bcicompet_raw_dict.pkl', 'rb') as file:
                loaded_dict = pickle.load(file)
        else:
            EEG_data = {}
            if target_subjects == "all":
                target_subjects = ["A01","A02","A03","A04","A05","A06","A07","A08","A09"]

            path = "D:\\BCI_Competition_2a\\Training\\"
            list_dir = listdir(path)

            for i in range (0,len(target_subjects)):
                raw_gdf = mne.io.read_raw_gdf(path+list_dir[i],preload = False, verbose = False)
                EEG_data[target_subjects[i]] = {"Raw_data": raw_gdf.copy().resample(self.fs)}

            print(f"Successful to create Data of {target_subjects}")

        return loaded_dict
    
    def GetEpoch(self, EEG_data, tmin=-2.0, tmax=6.0, crop=(0,2),baseline = (-0.5,0.0), band_pass = (6,32), trial_removal_th = 100):
        EEG_epoch = {}
        for key_subs in EEG_data:
            raw_edf = EEG_data[key_subs]["Raw_data"]

            events, event_dict = mne.events_from_annotations(raw_edf)

            if key_subs == 'A04':
                event_dict =  {'769': 5,
                '770': 6,
                '772': 8,
                '771': 7}
                mapping = {5: 0, 6: 1, 8: 2, 7: 3}
                selected_events = events[np.isin(events[:, 2], [5, 6, 7, 8])]

            else:
                event_dict =  {'769': 7,
                '770': 8,
                '772': 10,
                '771': 9}
                mapping = {7: 0, 8: 1, 10: 2, 9: 3}
                selected_events = events[np.isin(events[:, 2], [7, 8, 9, 10])]

            Epochs = mne.Epochs(raw_edf, selected_events, 
                tmin= tmin,  
                tmax= tmax,    
                event_id=event_dict,
                preload = True,
                event_repeated='drop',
                baseline=baseline,
                verbose=False
                )
            
            EEG_epoch[key_subs] =  {"Raw_Epoch": Epochs.copy().pick(self.picks).crop(tmin= crop[0], tmax= crop[1])}
            train_data = EEG_epoch[key_subs]['Raw_Epoch'].copy().get_data() * 10e5 
            labels = EEG_epoch[key_subs]["Raw_Epoch"].copy().events[:,-1]

            labels = np.vectorize(mapping.get)(labels) #apply mapping label

            outlier_trial = []
            for ii in range(0,train_data.shape[0]):
                if train_data[ii].max() > trial_removal_th or train_data[ii].min() < -trial_removal_th:
                    outlier_trial.append(ii)
                    print(key_subs,train_data[ii].min(), ii)
                    print(key_subs,train_data[ii].max(), ii)

            EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(train_data, outlier_trial, axis = 0)
            EEG_epoch[key_subs]['label'] = np.delete(labels, outlier_trial)

            filtered_data = self.butter_bandpass_filter(EEG_epoch[key_subs]['Raw_Epoch'], lowcut= band_pass[0], highcut= band_pass[1])
            EEG_epoch[key_subs]['Raw_Epoch'] = filtered_data

            #Classes selection
            if "Left" not in self.allclass:
                EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(EEG_epoch[key_subs]['Raw_Epoch'], np.where(EEG_epoch[key_subs]['label']== 0), axis = 0)
                EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], np.where(EEG_epoch[key_subs]['label']== 0))

            if "Right" not in self.allclass:
                EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(EEG_epoch[key_subs]['Raw_Epoch'], np.where(EEG_epoch[key_subs]['label']== 1), axis = 0)
                EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], np.where(EEG_epoch[key_subs]['label']== 1))

            if "Non" not in self.allclass:
                EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(EEG_epoch[key_subs]['Raw_Epoch'], np.where(EEG_epoch[key_subs]['label']== 2), axis = 0)
                EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], np.where(EEG_epoch[key_subs]['label']== 2))

            if "Feet" not in self.allclass:
                EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(EEG_epoch[key_subs]['Raw_Epoch'], np.where(EEG_epoch[key_subs]['label']== 3), axis = 0)
                EEG_epoch[key_subs]['label'] = np.delete(EEG_epoch[key_subs]['label'], np.where(EEG_epoch[key_subs]['label']== 3))

        return EEG_epoch