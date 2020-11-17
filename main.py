from flask import Flask, render_template, request
import datetime
from random import randint, uniform
import json
from colorama import Fore, Back, Style
from colorama import init
init()

from condition_monitoring import condition_monitoring as cm

# import os
# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# # import seaborn as sns
# # sns.set(color_codes=True)
# # import matplotlib.pyplot as plt

# from numpy.random import seed
# import tensorflow as tf

# from keras.layers import Input, Dropout
# from keras.layers.core import Dense 
# from keras.models import Model, Sequential, load_model
# from keras import regularizers
# from keras.models import model_from_json



# app = Flask(__name__)

# @app.route("/")
# def home():
#       preprocessing()
#       return render_template("templates/home.html")

if __name__ == "__main__":
      
      anomaly_detection = cm.autoencoder()
      # for i in anomaly_detection.anomaly_detected['anomaly_detected']:
      #       if(i):
      #             print(Back.RED + 'ANOMALY')
      #       else:
      #             print(Back.GREEN + 'NORMAL')

      # app.run(host='192.168.146.1', port=80, debug=True)

      # with open('[ENV] merged_dataset_BearingTest_2.csv','r') as t1, open('merged_dataset_BearingTest_2.csv','r') as t2:
      #       fileone = t1.readlines()
      #       filetwo = t2.readlines()

      # with open('update.csv', 'w') as outFile:
      #       for line in filetwo:
      #             if line not in fileone:
      #                   outFile.write(line)   

      # t1.close(), t2.close(), outFile.close()