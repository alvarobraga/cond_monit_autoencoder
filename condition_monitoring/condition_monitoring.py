import tensorflow as tf
from keras.models import load_model
import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
import matplotlib.pyplot as plt

class autoencoder:

    def __init__(self):
        path2model = 'C:/Users/alvar/Machine_learning_for_anomaly_detection_and_condition_monitoring/env2/model'
        self.anomaly_detected = None      
        self.__model = load_model(path2model)
        self.__data_loading()

    
    def __data_loading(self):
        dataset_dir = "C:/Users/alvar/Machine_learning_for_anomaly_detection_and_condition_monitoring/env2/dataset"
        self.__merged_data = pd.DataFrame()

        for filename in os.listdir(dataset_dir):
            dataset=pd.read_csv(os.path.join(dataset_dir, filename), sep='\t')
            dataset_mean_abs = np.array(dataset.abs().mean())
            dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
            dataset_mean_abs.index = [filename]
            self.__merged_data = self.__merged_data.append(dataset_mean_abs)

        self.__merged_data.columns = ['Bearing 1','Bearing 2','Bearing 3','Bearing 4']
        self.__merged_data.index = pd.to_datetime(self.__merged_data.index, format='%Y.%m.%d.%H.%M.%S')
        self.__merged_data = self.__merged_data.sort_index()
        # self.__merged_data.to_csv('merged_dataset_BearingTest_2.csv')
        # print(self.__merged_data.head(20))
        # self.__a = self.__merged_data['2004-02-12 11:02:39':'2004-02-13 23:52:39']
        # self.__merged_data = self.__merged_data['2004-02-13 23:52:39':'2004-02-14 23:52:39']
                                           
        self.__normalize_data()


    def __normalize_data(self):
        a = self.__merged_data['2004-02-12 11:02:39':'2004-02-13 23:52:39']

        scaler = preprocessing.MinMaxScaler()
        scaler.fit_transform(a)

        b = self.__merged_data['2004-02-13 23:52:39':]

        # b = pd.DataFrame(scaler.fit_transform(a), 
        #                            columns=self.__a.columns, 
        #                            index=self.__a.index
        #                            )

        self.__data = pd.DataFrame(scaler.transform(b), 
                                   columns=b.columns, 
                                   index=b.index
                                   )        

        # self.__data = pd.DataFrame(scaler.transform(self.__merged_data), 
        #                            columns=self.__merged_data.columns, 
        #                            index=self.__merged_data.index
        #                            )
        self.__detect_anomaly()
    

    def __detect_anomaly(self):
        pred = self.__model.predict(np.array(self.__data))
        pred = pd.DataFrame(pred,
                            columns=self.__data.columns
                            )
        pred.index = self.__data.index
        scored = pd.DataFrame(index=self.__data.index)
        scored['Loss_mae'] = np.mean(np.abs(pred - self.__data), axis = 1)
        scored['Threshold'] = .25
        scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
        self.anomaly_detected = scored['Anomaly']
        self.anomaly_detected = {'Loss_mae':scored['Loss_mae'],
                                 'anomaly_detected':scored['Anomaly']
                                }

        # print(scored['Loss_mae'].head(20))
        scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','red'])
        plt.show()