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

class Unicorn:
    def __init__(self, selectclass, desired_fz, ch_pick):
        self.allclass = selectclass
        self.fs = desired_fz
        self.picks = ch_pick
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

    def GetRawEDF(self, target_subjects=["pipo","NutF8","AJpang","Aoomim","voen"], condition="offline"):
        EEG_data = {}

        if target_subjects == "all":
            target_subjects = ["pipo","NutF8","AJpang","Aoomim","voen","pipo_HCI","Kawin"]

        for i in range (0,len(target_subjects)):

            path = "C:\\git\Senior_Thesis\\DataSet\\"+condition+"\\"+ target_subjects[i] +"\\notch_EDF\\"
            list_dir = listdir(path)
            raw_each = [0] * len(list_dir)
            for j in range(len(list_dir)):
                raw_each[j] = mne.io.read_raw_edf(path+list_dir[j], preload = False, verbose=False)
                
            raw_edf = mne.concatenate_raws(raw_each)

            eegbci.standardize(raw_edf)  # set channel names
            montage = make_standard_montage("standard_1005")
            raw_edf.set_montage(montage)

            EEG_data[target_subjects[i]] = {"Raw_data": raw_edf.copy()}
            self.ch_names = EEG_data[target_subjects[i]]['Raw_data'].ch_names

            #Resample
            if self.fs != 250:
                for key in EEG_data.keys():
                    EEG_data[key]['Raw_data'] = EEG_data[key]['Raw_data'].resample(128) 

        print(f"Successful to create Data of {target_subjects}")
        return EEG_data
    
    def GetEpoch(self, EEG_data, tmin=-2.0, tmax=6.0, crop=(0,4) ,baseline = (-0.5,0.0), band_pass = (6,32),trial_removal_th = 100):
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
            filtered_data = self.butter_bandpass_filter(EEG_epoch[key_subs]['Raw_Epoch'], lowcut= band_pass[0], highcut= band_pass[1])
            EEG_epoch[key_subs]['Raw_Epoch'] = filtered_data

            #select class
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
    
    def ComputeEA(self, Epochs_data, target_subject = 'pipo', calibrate_size = 0.2):
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
                # Calculate reference matrix
                RefEA = 0
                # Iterate over all trials, compute reference EA
                for trial in Epochs_data[key_subs]["Raw_Epoch"]:
                    cov = np.cov(trial)
                    RefEA += cov
                # Average over all trials
                RefEA = RefEA/Epochs_data[key_subs]["Raw_Epoch"].shape[0]

                # Compute R^(-0.5)
                R_inv = sqrtm(inv(RefEA))
                
                EA_data = []
                for trial in Epochs_data[key_subs]["Raw_Epoch"]:
                    EA_data.append(R_inv@trial)

                Epochs_data[key_subs]['EA_Epoch'] = np.array(EA_data)

                if (key_subs == target_subject) and calibrate_size != 0: #Apply RefEA to target_test data
                    R_inv = sqrtm(inv(RefEA))
                    EA_data = []
                    for trial in Epochs_data[tgt_test]["Raw_Epoch"]:
                        EA_data.append(R_inv@trial)
                    Epochs_data[tgt_test]['EA_Epoch'] = np.array(EA_data)


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
        lda.fit(X_train, train_label)
        self.GetConfusionMatrix(lda, X_train, X_test, train_label, test_label)


    def classifyCSP_SVM(self, data, target_subjects, calibrate_data,condition = "noEA"):
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
        cv_splitter = KFold(n_splits=2, shuffle=True, random_state=42)
        tuned_clf = GridSearchCV(estimator=svm, param_grid=param_grid,
                         scoring='accuracy', refit='accuracy', cv=cv_splitter)
        
        tuned_clf.fit(X_train, train_label)

        print(f"Best parameters: {tuned_clf.best_params_}")
        print(f"Best cross-validation score: {tuned_clf.best_score_:.3f}")

        self.GetConfusionMatrix(tuned_clf, X_train, X_test, train_label, test_label)


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