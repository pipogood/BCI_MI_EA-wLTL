import mne
import numpy as np
from mne.datasets import eegbci
import matplotlib.pyplot as plt
from os import listdir
from mne.channels import make_standard_montage
from scipy import signal
from scipy.linalg import sqrtm, inv 
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.utils import shuffle
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit,StratifiedKFold ,cross_val_score, cross_val_predict, KFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from pyriemann.utils.distance import distance_riemann
from scipy.linalg import logm, expm


class BCIFuntions:
    def __init__(self, numclass, frequency, ch_pick):
        self.numclass = numclass
        self.fs = frequency
        self.picks = ch_pick
    
    def GetRawEDF(self, target_subjects="pipo", condition="offline"):
        EEG_data = {}

        if condition == "offline":
            condition = "Offline_Experiment"
        elif condition == "online":
            condition = "Online_Experiment"

        if target_subjects == "all":
            target_subjects = ["pipo","NutF8","AJpang","Aoomim","voen"]

        for i in range (0,len(target_subjects)):

            path = "C:\\git\Senior_Thesis\\DataSet\\"+condition+"\\"+ target_subjects[i] +"\\notch_EDF\\"
            list_dir = listdir(path)
            raw_each = [0] * len(list_dir)
            for j in range(len(list_dir)):
                raw_each[j] = mne.io.read_raw_edf(path+list_dir[j],preload = False)
                
            raw_edf = mne.concatenate_raws(raw_each)

            eegbci.standardize(raw_edf)  # set channel names
            montage = make_standard_montage("standard_1005")
            raw_edf.set_montage(montage)

            EEG_data[target_subjects[i]] = {"Raw_data": raw_edf.copy()}

            self.ch_names = EEG_data[target_subjects[i]]['Raw_data'].ch_names

        print(f"Successful to create Data of {target_subjects}")

        return EEG_data
    
    def GetEpoch(self, EEG_data, tmin=-2.0, tmax=6.0, crop=(0,2),baseline = (-0.5,0.0), trial_removal_th = 100):

        EEG_epoch = {}

        for key_subs in EEG_data:
            raw_edf = EEG_data[key_subs]["Raw_data"]

            events, event_dict = mne.events_from_annotations(raw_edf)

            event_dict =  {'OVTK_GDF_Left': 2,
            'OVTK_GDF_Right': 3,
            'OVTK_GDF_Tongue': 4,
            'OVTK_GDF_Up': 5}

            events_1 = np.delete(events, [0], axis= 0)
            arr2= np.arange(len(events_1))
            events = events_1[(arr2 % 5 == 0)]

            Epochs = mne.Epochs(raw_edf, events, 
                tmin= tmin,  
                tmax= tmax,    
                event_id=event_dict,
                preload = True,
                event_repeated='drop',
                baseline=baseline,
                verbose=False
                )
            
            EEG_epoch[key_subs] =  {"Raw_Epoch": Epochs.copy().pick(self.picks).crop(tmin= crop[0], tmax= crop[1])}

            train_data = EEG_epoch[key_subs]['Raw_Epoch'].get_data()
            labels = EEG_epoch[key_subs]["Raw_Epoch"].copy().events[:,-1]

            mapping = {2: 0, 3: 1, 4: 2, 5: 3}
            labels = np.vectorize(mapping.get)(labels)

            outlier_trial = []
            for ii in range(0,train_data.shape[0]):
                if train_data[ii].max() > trial_removal_th or train_data[ii].min() < -trial_removal_th:
                    outlier_trial.append(ii)
                    print(key_subs,train_data[ii].min(), ii)
                    print(key_subs,train_data[ii].max(), ii)

            EEG_epoch[key_subs]['Raw_Epoch'] = np.delete(train_data, outlier_trial, axis = 0)
            EEG_epoch[key_subs]['label'] = np.delete(labels, outlier_trial)

            #apply filter
            filtered_data = self.butter_bandpass_filter(EEG_epoch[key_subs]['Raw_Epoch'], lowcut= 6, highcut= 32)
            EEG_epoch[key_subs]['Raw_Epoch'] = filtered_data

        return EEG_epoch

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
    

    def GetRawSet_ComputeEA(self, Epochs_data, target_subject = 'pipo', calibrate_size = 0.2):
        '''
        Note: EEG labels 
        class 0 is Left hand imagery
        class 1 is Right hand imagery
        class 3 is Feet imagery
        class 2 is Non-imagery

        For motor imagery, we will crop data at 0-4 second after direction cue represent and filter 6-32 Hz

        This function will return raw_EEG and EA_EEG
        '''
        label_target = Epochs_data[target_subject]["label"]

        if calibrate_size != 0:
            x_train, x_test, y_train, y_test = train_test_split(Epochs_data[target_subject]['Raw_Epoch'], label_target, test_size=calibrate_size, random_state = 42, stratify=label_target)
            
            tgt_test = str(target_subject) + "_test"
            Epochs_data[tgt_test] = {"Raw_Epoch": x_train}
            Epochs_data[tgt_test]['label'] = y_train

            Epochs_data[target_subject]["label"] = y_test
            Epochs_data[target_subject]["Raw_Epoch"] = x_test

        else:
            tgt_test = target_subject
            Epochs_data[target_subject]['EA_Epoch'] = Epochs_data[target_subject]['Raw_Epoch']

        for key_subs in Epochs_data:
            if key_subs != tgt_test:

                label = Epochs_data[key_subs]["label"]
                Epochs_data[key_subs]["Raw_left"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 0)]
                Epochs_data[key_subs]["Raw_right"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 1)]
                Epochs_data[key_subs]["Raw_non"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 2)]
                Epochs_data[key_subs]["Raw_feet"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 3)]
                # Calculate reference matrix
                RefEA = 0
                # Iterate over all trials, compute reference EA
                for trial in Epochs_data[key_subs]["Raw_Epoch"]:
                    cov = np.cov(trial)
                    RefEA += cov
                # Average over all trials
                RefEA = RefEA/Epochs_data[key_subs]["Raw_Epoch"].shape[0]

                EA_left = []
                EA_right = []
                EA_feet = []
                EA_non = []
                # Compute R^(-0.5)
                R_inv = sqrtm(inv(RefEA))
                
                for left, right, feet, non in zip(Epochs_data[key_subs]["Raw_left"] , Epochs_data[key_subs]["Raw_right"] , Epochs_data[key_subs]["Raw_feet"] , Epochs_data[key_subs]["Raw_non"] ):
                    EA_left.append(R_inv@left)
                    EA_right.append(R_inv@right)
                    EA_feet.append(R_inv@feet)
                    EA_non.append(R_inv@non)
                # Store as left_EA and right_EA
                Epochs_data[key_subs]['EA_left'] = np.array(EA_left)
                Epochs_data[key_subs]['EA_right'] = np.array(EA_right)
                Epochs_data[key_subs]['EA_feet'] = np.array(EA_feet)
                Epochs_data[key_subs]['EA_non'] = np.array(EA_non)
                EA_data = []
                for trial in Epochs_data[key_subs]["Raw_Epoch"]:
                    EA_data.append(R_inv@trial)

                Epochs_data[key_subs]['EA_Epoch'] = np.array(EA_data)

                if (key_subs == target_subject) and calibrate_size != 0: #Apply RefEA to target_test data
                    label = Epochs_data[tgt_test]["label"]
                    Epochs_data[tgt_test]["Raw_left"] = Epochs_data[tgt_test]["Raw_Epoch"][np.where(label == 0)]
                    Epochs_data[tgt_test]["Raw_right"] = Epochs_data[tgt_test]["Raw_Epoch"][np.where(label == 1)]
                    Epochs_data[tgt_test]["Raw_non"] = Epochs_data[tgt_test]["Raw_Epoch"][np.where(label == 2)]
                    Epochs_data[tgt_test]["Raw_feet"] = Epochs_data[tgt_test]["Raw_Epoch"][np.where(label == 3)]
                    EA_left = []
                    EA_right = []
                    EA_feet = []
                    EA_non = []
                    # Compute R^(-0.5)
                    R_inv = sqrtm(inv(RefEA))
                    for left, right, feet, non in zip(Epochs_data[tgt_test]["Raw_left"] , Epochs_data[tgt_test]["Raw_right"] , Epochs_data[tgt_test]["Raw_feet"] , Epochs_data[tgt_test]["Raw_non"] ):
                        EA_left.append(R_inv@left)
                        EA_right.append(R_inv@right)
                        EA_feet.append(R_inv@feet)
                        EA_non.append(R_inv@non)
                    # Store as left_EA and right_EA
                    Epochs_data[tgt_test]['EA_left'] = np.array(EA_left)
                    Epochs_data[tgt_test]['EA_right'] = np.array(EA_right)
                    Epochs_data[tgt_test]['EA_feet'] = np.array(EA_feet)
                    Epochs_data[tgt_test]['EA_non'] = np.array(EA_non)
                    EA_data = []
                    for trial in Epochs_data[tgt_test]["Raw_Epoch"]:
                        EA_data.append(R_inv@trial)
                    Epochs_data[tgt_test]['EA_Epoch'] = np.array(EA_data)

                    
    def plot_rawEA(self, Epochs_data, no_trial = 0, target_subject = "pipo"):
        '''
        This function will plot time series data compare between EA and non-EA 
        Require raw_data, EA_data in a list of 4 classes and selected trial to plot
        '''

        left = Epochs_data[target_subject]['Raw_left'][no_trial]
        right = Epochs_data[target_subject]['Raw_right'][no_trial]
        feet = Epochs_data[target_subject]['Raw_feet'][no_trial]
        noim = Epochs_data[target_subject]['Raw_non'][no_trial]

        left_alg = Epochs_data[target_subject]['EA_left'][no_trial]
        right_alg = Epochs_data[target_subject]['EA_right'][no_trial]
        feet_alg = Epochs_data[target_subject]['EA_feet'][no_trial]
        non_alg = Epochs_data[target_subject]['EA_non'][no_trial]

        # Time
        time = np.arange(Epochs_data[target_subject]['Raw_left'].shape[2])/self.fs #total time

        # Number of electrodes
        n_el = Epochs_data[target_subject]['Raw_left'].shape[1]

        # Plot each raw EEG and aligned raw EEG
        fig, axes = plt.subplots(n_el, self.numclass, figsize=(20, 20), sharex=True)

        # Title to each column
        axes[0, 0].set_title('Left')
        axes[0, 1].set_title('Right')
        axes[0, 2].set_title('Feet')
        axes[0, 3].set_title('Non-imagine')

        # Plot each eeg raw and aligned 
        for i, ax in enumerate(axes):
            ax[0].plot(time, left[i], color='k', label= str(self.ch_names[i]))
            ax[0].plot(time, left_alg[i], color='r', label= str(self.ch_names[i]))
            ax[0].legend(loc=1)
            
            ax[1].plot(time, right[i], color='k', label= str(self.ch_names[i]))
            ax[1].plot(time, right_alg[i], color='r', label= str(self.ch_names[i]))
            ax[1].legend(loc=1)

            ax[2].plot(time, feet[i], color='k', label= str(self.ch_names[i]))
            ax[2].plot(time, feet_alg[i], color='r', label= str(self.ch_names[i]))
            ax[2].legend(loc=1)
            
            ax[3].plot(time, noim[i], color='k', label= str(self.ch_names[i]))
            ax[3].plot(time, non_alg[i], color='r', label= str(self.ch_names[i]))
            ax[3].legend(loc=1)

        plt.show()

    def k_medoids_clustering_train(self, cov_matrices, k, label):
        distances = np.zeros((len(cov_matrices), len(cov_matrices)))
        for i in range(len(cov_matrices)):
            for j in range(len(cov_matrices)):
                distances[i, j] = distance_riemann(cov_matrices[i], cov_matrices[j])
        knn = KNeighborsClassifier(n_neighbors=k, metric='precomputed', weights= 'distance')
        knn.fit(distances, label)
        print(knn.score(distances, label))
        return knn

    def k_medoids_clustering_test(self, cov_tgt, cov_target, model):
        distances = np.zeros((len(cov_tgt), len(cov_target)))
        for i in range(len(cov_tgt)):
            for j in range(len(cov_target)): 
                distances[i, j] = distance_riemann(cov_tgt[i], cov_target[j])
        labels = model.predict(distances) 
        return labels
    
    def log_euclidean_mean(self, matrices):
        log_matrices = [logm(matrix) for matrix in matrices]
        mean_log_matrix = np.mean(log_matrices, axis=0)
        mean_matrix = expm(mean_log_matrix)

        return mean_matrix
    
    def ComputeLA(self, Epochs_data, target_subject = 'pipo', calibrate_size = 0.2):
        label_target = Epochs_data[target_subject]["label"]
        x_train, x_test, y_train, y_test = train_test_split(Epochs_data[target_subject]['Raw_Epoch'], label_target, test_size=calibrate_size, random_state = 42, stratify=label_target)
        
        tgt_test = str(target_subject) + "_test"
        Epochs_data[tgt_test] = {"Raw_Epoch": x_train}
        Epochs_data[tgt_test]['label'] = y_train

        Epochs_data[target_subject]["label"] = y_test
        Epochs_data[target_subject]["Raw_Epoch"] = x_test

        source_left = []
        source_right = []
        source_feet = []
        source_non = []

        for key_subs in Epochs_data:
            if key_subs != tgt_test and key_subs != target_subject:
                print("source; ",key_subs)
                label = Epochs_data[key_subs]["label"]
                Epochs_data[key_subs]["Raw_left"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 0)]
                Epochs_data[key_subs]["Raw_right"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 1)]
                Epochs_data[key_subs]["Raw_non"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 2)]
                Epochs_data[key_subs]["Raw_feet"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 3)]
                source_left.append(Epochs_data[key_subs]["Raw_left"])
                source_right.append(Epochs_data[key_subs]["Raw_right"])
                source_feet.append(Epochs_data[key_subs]["Raw_feet"])
                source_non.append(Epochs_data[key_subs]["Raw_non"])
                    
            elif key_subs == target_subject:
                target_cov = []
                global knn_model
                global csp
                print("target; ",target_subject)
                label = Epochs_data[key_subs]["label"]
                Epochs_data[key_subs]["Raw_left"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 0)]
                Epochs_data[key_subs]["Raw_right"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 1)]
                Epochs_data[key_subs]["Raw_non"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 2)]
                Epochs_data[key_subs]["Raw_feet"] = Epochs_data[key_subs]["Raw_Epoch"][np.where(label == 3)]

                for trial in Epochs_data[key_subs]["Raw_Epoch"]:
                    target_cov.append(np.cov(trial))
                knn_model = self.k_medoids_clustering_train(target_cov, 5, label)

                Ctm_left = np.cov(np.mean(Epochs_data[key_subs]["Raw_left"], axis = 0))
                Ctm_right = np.cov(np.mean(Epochs_data[key_subs]["Raw_right"], axis = 0))
                Ctm_feet = np.cov(np.mean(Epochs_data[key_subs]["Raw_feet"], axis = 0))
                Ctm_non = np.cov(np.mean(Epochs_data[key_subs]["Raw_non"], axis = 0))

                # Ctm_left = []
                # Ctm_right = [] 
                # Ctm_feet = []
                # Ctm_non = []

                # for trial in Epochs_data[key_subs]["Raw_left"]:
                #     Ctm_left.append(np.cov(trial))
                # Ctm_left = self.log_euclidean_mean(np.array(Ctm_left))

                # for trial in Epochs_data[key_subs]["Raw_right"]:
                #     Ctm_right.append(np.cov(trial))
                # Ctm_right = self.log_euclidean_mean(np.array(Ctm_right))

                # for trial in Epochs_data[key_subs]["Raw_feet"]:
                #     Ctm_feet.append(np.cov(trial))
                # Ctm_feet = self.log_euclidean_mean(np.array(Ctm_feet))

                # for trial in Epochs_data[key_subs]["Raw_non"]:
                #     Ctm_non.append(np.cov(trial))
                # Ctm_non = self.log_euclidean_mean(np.array(Ctm_non))


            elif key_subs == tgt_test:
                tgt_cov = []
                for trial in Epochs_data[key_subs]["Raw_Epoch"]:
                    tgt_cov.append(np.cov(trial))
                tgt_label = self.k_medoids_clustering_test(tgt_cov, target_cov, knn_model)


                Epochs_data[key_subs]["KMediod_label"] = tgt_label

                
        Csm_left = np.cov(np.mean(np.concatenate(source_left, axis = 0), axis = 0))
        Csm_right = np.cov(np.mean(np.concatenate(source_right, axis = 0), axis = 0))
        Csm_feet = np.cov(np.mean(np.concatenate(source_feet, axis = 0), axis = 0))
        Csm_non = np.cov(np.mean(np.concatenate(source_non, axis = 0), axis = 0))

        # Csm_left_0 = np.concatenate(source_left, axis = 0)
        # Csm_right_0 = np.concatenate(source_right, axis = 0)
        # Csm_feet_0 = np.concatenate(source_feet, axis = 0)
        # Csm_non_0 = np.concatenate(source_non, axis = 0)
        # Csm_left = []
        # Csm_right = [] 
        # Csm_feet = []
        # Csm_non = []
        # for trial in Csm_left_0:
        #     Csm_left.append(np.cov(trial))
        # Csm_left = self.log_euclidean_mean(np.array(Csm_left))
        
        # for trial in Csm_right_0:
        #     Csm_right.append(np.cov(trial))
        # Csm_right = self.log_euclidean_mean(np.array(Csm_right))

        # for trial in Csm_feet_0:
        #     Csm_feet.append(np.cov(trial))
        # Csm_feet = self.log_euclidean_mean(np.array(Csm_feet))

        # for trial in Csm_non_0:
        #     Csm_non.append(np.cov(trial))
        # Csm_non = self.log_euclidean_mean(np.array(Csm_non))
        

        Am_left = np.dot(np.linalg.cholesky(Ctm_left), np.linalg.inv(np.linalg.cholesky(Csm_left)))
        Am_right = np.dot(np.linalg.cholesky(Ctm_right), np.linalg.inv(np.linalg.cholesky(Csm_right)))
        Am_feet = np.dot(np.linalg.cholesky(Ctm_feet), np.linalg.inv(np.linalg.cholesky(Csm_feet)))
        Am_non = np.dot(np.linalg.cholesky(Ctm_non), np.linalg.inv(np.linalg.cholesky(Csm_non)))


        for key_subs in Epochs_data:
            EA_data = []
            
            if key_subs != tgt_test:
                label = Epochs_data[key_subs]["label"]
            else:
                label = Epochs_data[key_subs]["KMediod_label"]

            for i, trial in enumerate(Epochs_data[key_subs]["Raw_Epoch"]):
                if np.isin(i, np.where(label == 0)):
                    EA_data.append(Am_left@trial)
                elif np.isin(i, np.where(label == 1)):
                    EA_data.append(Am_right@trial)
                elif np.isin(i, np.where(label == 2)):
                    EA_data.append(Am_non@trial)
                elif np.isin(i, np.where(label == 3)):
                    EA_data.append(Am_feet@trial)
                    

            Epochs_data[key_subs]['EA_Epoch'] = np.array(EA_data)


    def computeCSPFeatures(self, data, target_subject = "pipo_test", target_subject_0 = "pipo"):

        train_data = None
        train_label = None

        CSP_Epoch = {} 
        for sub in data.keys():
            CSP_Epoch[sub] = {}

        conditions = ["noEA", "EA"]

        for condition in conditions:
            if condition == "noEA":
                query = "Raw_Epoch"
            else:
                query = "EA_Epoch"

            for sub in data.keys():
                if sub == target_subject or sub == target_subject_0:
                    pass
                else:
                    if train_data is None:
                        train_data = data[sub][query]
                    else:
                        train_data = np.concatenate((train_data, data[sub][query]), axis=0)

                    if train_label is None:
                        train_label = data[sub]['label']
                    else:
                        train_label = np.concatenate((train_label, data[sub]['label']), axis=0)

            
            csp = CSP(n_components = len(self.picks), reg=None, log=None, rank= 'info')

            train_data, train_label = shuffle(train_data, train_label, random_state = 0)
            csp.fit(train_data, train_label)      
            
            for key_sub in data:
                if condition == "noEA":
                    CSP_Epoch[key_sub]['Raw_csp'] = csp.transform(data[key_sub]['Raw_Epoch'])
                    CSP_Epoch[key_sub]['Raw_csp_label'] = data[key_sub]['label']
                else:
                    CSP_Epoch[key_sub]['EA_csp'] = csp.transform(data[key_sub]['EA_Epoch'])
                    CSP_Epoch[key_sub]['EA_csp_label'] = data[key_sub]['label']

        return CSP_Epoch


    def TSNE_Plot(self, CSP_Epoch, target_subject = "voen"):
        # Perform sne on all subject
        for key_subs in CSP_Epoch:
            print('Processing' , key_subs)
            CSP_Epoch[key_subs]['sne'] = TSNE(perplexity= 10,n_iter=5000).fit_transform(CSP_Epoch[key_subs]['Raw_csp'])
            CSP_Epoch[key_subs]['sne_EA'] = TSNE(perplexity= 10,n_iter=5000).fit_transform(CSP_Epoch[key_subs]['EA_csp'])


        palette = np.array(sns.color_palette(n_colors=11))

        # Distribution of all subjects
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))

        count = 0

        for key_subs in CSP_Epoch:
            count+= 1
            ax0.set_title('No EA')
            ax0.scatter(CSP_Epoch[key_subs]['sne'][:, 0], CSP_Epoch[key_subs]['sne'][:, 1], lw=0, s=40, color=palette[count], label=key_subs)
            ax0.legend()
            ax0.axis('off')
            ax0.axis('tight')
            
            ax1.set_title('EA')
            ax1.scatter(CSP_Epoch[key_subs]['sne_EA'][:, 0], CSP_Epoch[key_subs]['sne_EA'][:, 1], lw=0, s=40, color=palette[count], label=key_subs)
            ax1.legend()
            ax1.axis('off')
            ax1.axis('tight')

        plt.show()

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))

        colorlist = {
            'tgt' : 'red',
            'src' : 'blue'
        }

        for key_subs in CSP_Epoch:
            ax0.set_title('No EA')
            if key_subs == target_subject:
                ax0.scatter(CSP_Epoch[key_subs]['sne'][:, 0], CSP_Epoch[key_subs]['sne'][:, 1], lw=0, s=40, color=colorlist['tgt'], label='target')
            else:
                ax0.scatter(CSP_Epoch[key_subs]['sne'][:, 0], CSP_Epoch[key_subs]['sne'][:, 1], lw=0, s=40, color=colorlist['src'], label='source')
            ax0.legend()
            ax0.axis('off')
            ax0.axis('tight')
            
            ax1.set_title('with EA')
            if key_subs == target_subject:
                ax1.scatter(CSP_Epoch[key_subs]['sne_EA'][:, 0], CSP_Epoch[key_subs]['sne_EA'][:, 1], lw=0, s=40, color=colorlist['tgt'], label='target')
            else:
                ax1.scatter(CSP_Epoch[key_subs]['sne_EA'][:, 0], CSP_Epoch[key_subs]['sne_EA'][:, 1], lw=0, s=40, color=colorlist['src'], label='source')
            ax1.legend()
            ax1.axis('off')
            ax1.axis('tight')

        y = CSP_Epoch[target_subject]['Raw_csp_label']

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=CSP_Epoch[target_subject]['sne'][:, 0], y=CSP_Epoch[target_subject]['sne'][:, 1], hue=y, palette="deep")
        plt.title("Before EA of " + str(target_subject))
        plt.show()

        y = CSP_Epoch[target_subject]['EA_csp_label']

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=CSP_Epoch[target_subject]['sne_EA'][:, 0], y=CSP_Epoch[target_subject]['sne_EA'][:, 1], hue=y, palette="deep")
        plt.title("After EA of " + str(target_subject))
        plt.show()

    def classifyCSP_LDA(self, data, target_subjects, calibrate_data,condition = "noEA"):

        train_data = None
        train_label = None
        test_data = None
        test_label = None

        if condition == "noEA":
            query = "Raw_Epoch"
        else:
            query = "EA_Epoch"

        for sub in data.keys():
            if sub != calibrate_data:
                if sub == target_subjects:
                    test_data = data[sub][query]
                    test_label = data[sub]['label']

                else:
                    if train_data is None:
                        train_data = data[sub][query]
                    else:
                        train_data = np.concatenate((train_data, data[sub][query]), axis=0)

                    if train_label is None:
                        train_label = data[sub]['label']
                    else:
                        train_label = np.concatenate((train_label, data[sub]['label']), axis=0)

        
        csp = CSP(n_components = len(self.picks), reg=None, log=None, rank= 'info')

        train_data, train_label = shuffle(train_data, train_label, random_state = 0)
        test_data, test_label = shuffle(test_data, test_label, random_state = 0)

        csp.fit(train_data, train_label)

        X_train = csp.transform(train_data)
        X_test  = csp.transform(test_data)

        lda = LinearDiscriminantAnalysis()
        score = cross_val_score(lda, X_train, train_label, cv= 10)
        print("LDA only Cross-validation scores:", np.mean(score))
        lda.fit(X_train, train_label)
        self.GetConfusionMatrix(lda, X_train, X_test, train_label, test_label)


    def classifyCSP_SVM(self, data, target_subjects, condition = "noEA"):
        train_data = None
        train_label = None
        test_data = None
        test_label = None

        if condition == "noEA":
            query = "Raw_Epoch"
        else:
            query = "EA_Epoch"

        for sub in data.keys():
            if sub == target_subjects:
                test_data = data[sub][query]
                test_label = data[sub]['label']

            else:
                if train_data is None:
                    train_data = data[sub][query]
                else:
                    train_data = np.concatenate((train_data, data[sub][query]), axis=0)

                if train_label is None:
                    train_label = data[sub]['label']
                else:
                    train_label = np.concatenate((train_label, data[sub]['label']), axis=0)

        
        csp = CSP(n_components = len(self.picks) , reg=None, log=None, rank= 'info')

        train_data, train_label = shuffle(train_data, train_label, random_state = 0)
        test_data, test_label = shuffle(test_data, test_label, random_state = 0)

        csp.fit(train_data, train_label)

        X_train = csp.transform(train_data)
        X_test  = csp.transform(test_data)

        param_grid = {
            'C':  [1],
            'kernel': ['linear', 'rbf', 'poly']
        }
        
        svm =  SVC(random_state=42)
        cv_splitter = KFold(n_splits=10, shuffle=True, random_state=42)
        tuned_clf = GridSearchCV(estimator=svm, param_grid=param_grid,
                         scoring='accuracy', refit='accuracy', cv=cv_splitter)
        
        tuned_clf.fit(X_train, train_label)

        print(f"Best parameters: {tuned_clf.best_params_}")
        print(f"Best cross-validation score: {tuned_clf.best_score_:.3f}")

        self.GetConfusionMatrix(tuned_clf, X_train, X_test, train_label, test_label)


    def GetConfusionMatrix(self, models, X_train, X_test, y_train, y_test):
        y_pred = models.predict(X_train)
        print("Classification TRAIN DATA \n=======================")
        print(classification_report(y_true= y_train, y_pred=y_pred))
        print("Confusion matrix \n=======================")
        print(confusion_matrix(y_true= y_train, y_pred=y_pred))

        y_pred = models.predict(X_test)
        print("Classification TEST DATA \n=======================")
        print(classification_report(y_true=y_test, y_pred=y_pred))
        print("Confusion matrix \n=======================")
        print(confusion_matrix(y_true=y_test, y_pred=y_pred))