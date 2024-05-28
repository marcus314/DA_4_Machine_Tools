import augmentation_methods as aug
#import create_csv as cr_csv
import training_functions as trf
import os
import csv
import datetime
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import torch.jit
from joblib import Parallel, delayed

np.set_printoptions(floatmode='fixed')

WINDOWSIZE = 1 #1 for no additionalo smoothing
FILL_PROCESS_FOR_AIRCUT = False #True #only used if aircut&process are used together, e.g. Versuch 8a
PAST_VALUES = 2
FUTURE_VALUES = 2

####allchannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT = 0.02
CUTOFF = 4
current_directory = os.path.dirname(os.path.abspath(__file__))

TEST_RESULTS_PATH = current_directory+"\\Auswertung\\test_results.csv"
DEVIATION_LOG_PATH = current_directory+"\\Auswertung\\deviation_log.csv"
CURRENT_DIRECTORY = current_directory



#CLASSES THAT CONTAIN PARAMETERS FOR DATA. Should not be in use anymore, instead use data_Versuch_XY
class data_master:
    def __init__(self):
        self.trainig_datapaths = ['CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3']#['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3','CMX_Alu_Tr_Air_4']
        self.validation_datapaths = ['CMX_St_Val_Air_1','CMX_St_Val_Air_2','CMX_St_Val_Air_3','CMX_St_Val_Air_4']#['CMX_Alu_Val_Air_1','CMX_Alu_Val_Air_2','CMX_Alu_Val_Air_3','CMX_Alu_Val_Air_4']
        self.machine = 'CMX'
        self.training_channels = ['v_x', 'a_x']
        self.target_channels = ['cur_x']
        self.smoothing = 20
        self.modulo_split = 10

class data_Versuch_quicktest:
    def __init__(self):
        self.name = "Versuch_quicktest"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3']
        self.validation_datapaths = ['CMX_Alu_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

#Data der Versuche von Yannik
class data_Versuch_1_CMX_aircut:
    def __init__(self):
        self.name = "Versuch_1_CMX_aircut"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_3']
        self.validation_datapaths = ['CMX_St_Tr_Air_2']
        self.testing_datapaths = ['CMX_St_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_1_I40_aircut:
    def __init__(self):
        self.name = "Versuch_1_I40"
        self.trainig_datapaths = ['I40_Alu_Tr_Air_1','I40_Alu_Tr_Air_2','I40_Alu_Tr_Air_3','I40_St_Tr_Air_3']
        self.validation_datapaths = ['I40_St_Tr_Air_2']
        self.testing_datapaths = ['I40_St_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_1_CMX_prozess:
    def __init__(self):
        self.name = "Versuch_1_CMX_prozess"
        self.trainig_datapaths = ['CMX_Alu_Tr_Mat_1','CMX_Alu_Tr_Mat_2','CMX_Alu_Tr_Mat_3', 'CMX_Alu_Tr_Mat_4','CMX_St_Tr_Mat_3','CMX_St_Tr_Mat_4']
        self.validation_datapaths = ['CMX_St_Tr_Mat_2']
        self.testing_datapaths = ['CMX_St_Tr_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_1_I40_prozess:
    def __init__(self):
        self.name = "Versuch_1_I40_prozess"
        self.trainig_datapaths = ['I40_Alu_Tr_Mat_1','I40_Alu_Tr_Mat_2','I40_Alu_Tr_Mat_3','I40_St_Tr_Mat_3']
        self.validation_datapaths = ['I40_St_Tr_Mat_2']
        self.testing_datapaths = ['I40_St_Tr_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_2_CMX_aircut:
    def __init__(self):
        self.name = "Versuch_2_CMX_aircut"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3']
        self.validation_datapaths = ['CMX_Alu_Tr_Air_2']
        self.testing_datapaths = ['CMX_Alu_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_2_I40_aircut:
    def __init__(self):
        self.name = "Versuch_2_I40"
        self.trainig_datapaths = ['I40_Alu_Tr_Air_3','I40_St_Tr_Air_1','I40_St_Tr_Air_2','I40_St_Tr_Air_3']
        self.validation_datapaths = ['I40_Alu_Tr_Air_2']
        self.testing_datapaths = ['I40_Alu_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_1c:
    def __init__(self):
        self.name = "Versuch_1c"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3']
        self.validation_datapaths = ['CMX_Alu_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_2_CMX_prozess:
    def __init__(self):
        self.name = "Versuch_2_CMX_prozess"
        self.trainig_datapaths = ['CMX_Alu_Tr_Mat_3', 'CMX_Alu_Tr_Mat_4','CMX_St_Tr_Mat_1','CMX_St_Tr_Mat_2','CMX_St_Tr_Mat_3','CMX_St_Tr_Mat_4']
        self.validation_datapaths = ['CMX_Alu_Tr_Mat_2']
        self.testing_datapaths = ['CMX_Alu_Tr_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_2_I40_prozess:
    def __init__(self):
        self.name = "Versuch_2_I40_prozess"
        self.trainig_datapaths = ['I40_Alu_Tr_Mat_3','I40_St_Tr_Mat_1','I40_St_Tr_Mat_2','I40_St_Tr_Mat_3']
        self.validation_datapaths = ['I40_Alu_Tr_Mat_2']
        self.testing_datapaths = ['I40_Alu_Tr_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_3_CMX_aircut:
    def __init__(self):
        self.name = "Versuch_3_CMX_aircut"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4']
        self.validation_datapaths = ['CMX_St_Tr_Air_2']
        self.testing_datapaths = ['CMX_St_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_3_I40_aircut:
    def __init__(self):
        self.name = "Versuch_3_I40"
        self.trainig_datapaths = ['I40_Alu_Tr_Air_1','I40_Alu_Tr_Air_2','I40_Alu_Tr_Air_3']
        self.validation_datapaths = ['I40_St_Tr_Air_2']
        self.testing_datapaths = ['I40_St_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_2c:
    def __init__(self):
        self.name = "Versuch_2a"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4']
        self.validation_datapaths = ['CMX_St_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_3_CMX_prozess:
    def __init__(self):
        self.name = "Versuch_3_CMX_prozess"
        self.trainig_datapaths = ['CMX_Alu_Tr_Mat_1','CMX_Alu_Tr_Mat_2','CMX_Alu_Tr_Mat_3', 'CMX_Alu_Tr_Mat_4']
        self.validation_datapaths = ['CMX_St_Tr_Mat_2']
        self.testing_datapaths = ['CMX_St_Tr_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_3_I40_prozess:
    def __init__(self):
        self.name = "Versuch_3_I40_prozess"
        self.trainig_datapaths = ['I40_Alu_Tr_Mat_1','I40_Alu_Tr_Mat_2','I40_Alu_Tr_Mat_3']
        self.validation_datapaths = ['I40_St_Tr_Mat_2']
        self.testing_datapaths = ['I40_St_Tr_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_2b:
    def __init__(self):
        self.name = "Versuch_2b"
        self.trainig_datapaths = ['CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3']
        self.validation_datapaths = ['CMX_Alu_Tr_Air_1']
        self.target_channels = ['cur_x']

class data_Versuch_4_CMX_aircut:
    def __init__(self):
        self.name = "Versuch_4_CMX_aircut"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3']
        self.validation_datapaths = ['CMX_St_Val_Air_2']
        self.testing_datapaths = ['CMX_St_Val_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_4_I40_aircut:
    def __init__(self):
        self.name = "Versuch_4_I40"
        self.trainig_datapaths = ['I40_Alu_Tr_Air_1','I40_Alu_Tr_Air_2','I40_Alu_Tr_Air_3','I40_St_Tr_Air_1','I40_St_Tr_Air_2','I40_St_Tr_Air_3']
        self.validation_datapaths = ['I40_St_Val_Air_2']
        self.testing_datapaths = ['I40_St_Val_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_4_CMX_prozess:
    def __init__(self):
        self.name = "Versuch_4_CMX_prozess"
        self.trainig_datapaths = ['CMX_Alu_Tr_Mat_1','CMX_Alu_Tr_Mat_2','CMX_Alu_Tr_Mat_3', 'CMX_Alu_Tr_Mat_4','CMX_St_Tr_Mat_1', 'CMX_St_Tr_Mat_2','CMX_St_Tr_Mat_3','CMX_St_Tr_Mat_4']
        self.validation_datapaths = ['CMX_St_Val_Mat_2']
        self.testing_datapaths = ['CMX_St_Val_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_4_I40_prozess:
    def __init__(self):
        self.name = "Versuch_4_I40_prozess"
        self.trainig_datapaths = ['I40_Alu_Tr_Mat_1','I40_Alu_Tr_Mat_2','I40_Alu_Tr_Mat_3', 'I40_St_Tr_Mat_1', 'I40_St_Tr_Mat_2','I40_St_Tr_Mat_3']
        self.validation_datapaths = ['I40_St_Val_Mat_2']
        self.testing_datapaths = ['I40_St_Val_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_3b:
    def __init__(self):
        self.name = "Versuch_3b"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3']
        self.validation_datapaths = ['CMX_Alu_Val_Air_1']
        self.target_channels = ['cur_x']

class data_Versuch_4a:
    def __init__(self):
        self.name = "Versuch_4a"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3','CMX_Alu_Val_Air_2','CMX_Alu_Val_Air_3', 'CMX_Alu_Val_Air_4','CMX_St_Val_Air_2','CMX_St_Val_Air_3', 'CMX_St_Val_Air_4']
        self.validation_datapaths = ['CMX_St_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_5a:
    def __init__(self):
        self.name = "Versuch_5a"
        self.trainig_datapaths = ['CMX_Alu_Tr_Mat_2','CMX_Alu_Tr_Mat_3','CMX_Alu_Tr_Mat_4', 'CMX_St_Tr_Mat_2','CMX_St_Tr_Mat_3','CMX_St_Tr_Mat_4']
        self.validation_datapaths = ['CMX_St_Tr_Mat_1']
        self.target_channels = ['cur_y']

class data_Versuch_5b:
    def __init__(self):
        self.name = "Versuch_5b"
        self.trainig_datapaths = ['CMX_Alu_Tr_Mat_2','CMX_Alu_Tr_Mat_3','CMX_Alu_Tr_Mat_4', 'CMX_St_Tr_Mat_2','CMX_St_Tr_Mat_3','CMX_St_Tr_Mat_4']
        self.validation_datapaths = ['CMX_Alu_Tr_Mat_1']
        self.target_channels = ['cur_x']

class data_Versuch_5b_on_Paulas_Testbauteil4:
    def __init__(self):
        self.name = "Versuch_5b_on_Paulas_Testbauteil4"
        self.trainig_datapaths = ['CMX_Alu_Tr_Mat_2','CMX_Alu_Tr_Mat_3','CMX_Alu_Tr_Mat_4', 'CMX_St_Tr_Mat_2','CMX_St_Tr_Mat_3','CMX_St_Tr_Mat_4']
        self.validation_datapaths = ['Paulas_testbauteil4']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_8a:
    def __init__(self):
        self.name = "Versuch_8a"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3',
                                  'CMX_Alu_Val_Air_1','CMX_Alu_Val_Air_2','CMX_Alu_Val_Air_3', 'CMX_Alu_Val_Air_4','CMX_St_Val_Air_1','CMX_St_Val_Air_2','CMX_St_Val_Air_3','CMX_St_Val_Air_4',
                                  'CMX_Alu_Tr_Mat_1','CMX_Alu_Tr_Mat_2','CMX_Alu_Tr_Mat_3', 'CMX_Alu_Tr_Mat_4','CMX_St_Tr_Mat_1', 'CMX_St_Tr_Mat_2','CMX_St_Tr_Mat_3','CMX_St_Tr_Mat_4',
                                  'CMX_Alu_Val_Mat_1','CMX_Alu_Val_Mat_2','CMX_Alu_Val_Mat_3', 'CMX_Alu_Val_Mat_4',]
        self.validation_datapaths = ['CMX_St_Val_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_8b:
    def __init__(self):
        self.name = "Versuch_8b"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3',
                                  'CMX_Alu_Val_Air_1','CMX_Alu_Val_Air_2','CMX_Alu_Val_Air_3', 'CMX_Alu_Val_Air_4','CMX_St_Val_Air_1','CMX_St_Val_Air_2','CMX_St_Val_Air_3','CMX_St_Val_Air_4',
                                  'CMX_Alu_Tr_Mat_1','CMX_Alu_Tr_Mat_2','CMX_Alu_Tr_Mat_3', 'CMX_Alu_Tr_Mat_4','CMX_St_Tr_Mat_1', 'CMX_St_Tr_Mat_2','CMX_St_Tr_Mat_3','CMX_St_Tr_Mat_4',
                                  'CMX_St_Val_Mat_1','CMX_St_Val_Mat_2','CMX_St_Val_Mat_3', 'CMX_St_Val_Mat_4',]
        self.validation_datapaths = ['CMX_Alu_Val_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_8b_I40:
    def __init__(self):
        self.name = "Versuch_8b_I40"
        self.trainig_datapaths = ['I40_Alu_Tr_Air_1','I40_Alu_Tr_Air_2','I40_Alu_Tr_Air_3',
                                  'I40_St_Tr_Air_1','I40_St_Tr_Air_2','I40_St_Tr_Air_3',
                                  'I40_Alu_Val_Air_1','I40_Alu_Val_Air_2','I40_Alu_Val_Air_3',
                                  'I40_St_Val_Air_1','I40_St_Val_Air_2','I40_St_Val_Air_3',
                                  'I40_Alu_Tr_Mat_1','I40_Alu_Tr_Mat_2','I40_Alu_Tr_Mat_3',
                                  #'I40_Alu_Val_Mat_2','I40_Alu_Val_Mat_3',
                                  'I40_St_Tr_Mat_1', 'I40_St_Tr_Mat_2','I40_St_Tr_Mat_3',
                                  'I40_St_Val_Mat_1','I40_St_Val_Mat_2','I40_St_Val_Mat_3', 
                                  ]
        self.validation_datapaths = ['I40_Alu_Val_Mat_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class Versuch_4_I40:
    def __init__(self):
        self.name = "Versuch_4_I40"
        self.trainig_datapaths = ['I40_Alu_Tr_Air_2','I40_Alu_Tr_Air_3',
                                  'I40_St_Tr_Air_2','I40_St_Tr_Air_3',
                                  'I40_Alu_Val_Air_2','I40_Alu_Val_Air_3',
                                  'I40_St_Val_Air_2','I40_St_Val_Air_3',
                                  ]
        self.validation_datapaths = ['I40_Alu_Tr_Air_1']
        self.target_channels = ['cur_x']
        self.percentage_used = 100

class data_Versuch_1_for_MSEvalues:
    def __init__(self):
        self.name = "Versuch_1_for_MSEvalues"
        self.trainig_datapaths = ['CMX_Alu_Tr_Air_1','CMX_Alu_Tr_Air_2','CMX_Alu_Tr_Air_3', 'CMX_Alu_Tr_Air_4','CMX_St_Tr_Air_1','CMX_St_Tr_Air_2','CMX_St_Tr_Air_3']
        self.validation_datapaths = ['CMX_St_Tr_Air_1']
        self.target_channels = ['cur_x']
#CLASSES THAT CONTAIN PARAMETERS FOR AUGMENTATION
class aug_master:
    def __init__(self):
        self.augmentation1 = 'None'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class aug_Max:
    def __init__(self):
        self.augmentation1 = 'aug_Max'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class NoAugment:
    def __init__(self):
        self.augmentation1 = 'None'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class TwoTimesOriginal:
    def __init__(self):
        self.augmentation1 = 'TwoTimesOriginal'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class Homogenous_scaling:
    def __init__(self):
        self.augmentation1 = 'homogenous_scaling'
        self.aug1arg1 = 0.2 #Sigma
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class rotation:
    def __init__(self):
        self.augmentation1 = 'rotation'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class WindowWarpElong:
    def __init__(self):
        self.augmentation1 = 'windowWarpElong'
        self.aug1arg1 = 10
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class WindowWarpShort:
    def __init__(self):
        self.augmentation1 = 'windowWarpShort'
        self.aug1arg1 = 10
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
    

class AdaptiveNoise:
    def __init__(self):
        self.augmentation1 = 'AdaptiveNoise'
        self.aug1arg1 = 1 #factor for the stdev
        self.aug1arg2 = 100 #window length
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class AdaptiveNoiseSavgol:
    def __init__(self):
        self.augmentation1 = 'AdaptiveNoiseSavgol'
        self.aug1arg1 = 1 #factor for the stdev
        self.aug1arg2 = 9 #filter length
        self.aug1arg3 = 3 #polyorder of filter
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class smogn_augmentation:
    def __init__(self):
        self.augmentation1 = 'smogn_augmentation'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan 

class ydata_augmentation:
    def __init__(self):
        self.augmentation1 = 'ydata_augmentation'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan 

class TimeVAEgenerated:
    def __init__(self):
        self.augmentation1 = 'TimeVAEgenerated'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class TimeVAEgenerated_stacked:
    def __init__(self):
        self.augmentation1 = 'TimeVAEgenerated_stacked'
        self.aug1arg1 = 'stacked'
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class NoiseThenTimeWarp:
    def __init__(self):
        self.augmentation1 = 'Noise'
        self.aug1arg1 = 2 #Sigma
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'timeWarp'
        self.aug2arg1 = 4 #Knots
        self.aug2arg2 = 0.5 #Sigma
        self.aug2arg3 = 1 #mu
        self.aug2arg4 = 0.1 #min_dist
        self.aug2arg5 = np.nan

class TimeWarp_presentation:
    def __init__(self):
        self.augmentation1 = 'None'
        self.aug1arg1 = 5 #Knots
        self.aug1arg2 = 0.075 #Sigma
        self.aug1arg3 = 1 #mu
        self.aug1arg4 = 0.1 #min_dist
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class Overtrain_bad_MSEs:
    def __init__(self):
        self.augmentation1 = 'Overtrain_bad_MSEs'
        self.aug1arg1 = np.nan
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan

class LimitTrainingdata:
    def __init__(self):
        self.augmentation1 = 'LimitTrainingdata'
        self.aug1arg1 = 1 #percentage of data used for training
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan


#The methods used in my analysis
class MSE_threshold:
    def __init__(self):
        self.augmentation1 = 'MSE_threshold'
        self.aug1arg1 = 2 #Threshold (MSE/avg_MSE) over which value get doubled
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class MagnitudeWarp:
    def __init__(self):
        self.augmentation1 = 'magnitudeWarp'
        self.aug1arg1 = 5 #Knots
        self.aug1arg2 = 2 #Sigma
        self.aug1arg3 = 1 #mu
        self.aug1arg4 = 0.1 #min_dist
        self.aug1arg5 = np.nan
        self.percentage_used = 100
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class MagnitudeWarpThenTimeWarp:
    def __init__(self):
        self.augmentation1 = 'magnitudeWarp'
        self.aug1arg1 = 5 #Knots
        self.aug1arg2 = 2 #Sigma
        self.aug1arg3 = 1 #mu
        self.aug1arg4 = 0.1 #min_dist
        self.aug1arg5 = np.nan
        self.augmentation2 = 'timeWarp'
        self.aug2arg1 = 4 #Knots
        self.aug2arg2 = 0.5 #Sigma
        self.aug2arg3 = 1 #mu
        self.aug2arg4 = 0.1 #min_dist
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class Noise:
    def __init__(self):
        self.augmentation1 = 'Noise'
        self.aug1arg1 = 1 #Sigma
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class RandomDelete:
    def __init__(self):
        self.augmentation1 = 'RandomDelete'
        self.aug1arg1 = 5 #percentage of data NOT deleted
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class TimeWarp:
    def __init__(self):
        self.augmentation1 = 'timeWarp'
        self.aug1arg1 = 4 #Knots
        self.aug1arg2 = 0.5 #Sigma
        self.aug1arg3 = 0 #mu
        self.aug1arg4 = 0.1 #min_dist
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class TimeWarpThenNoise:
    def __init__(self):
        self.augmentation1 = 'timeWarpThenNoise'
        self.aug1arg1 = 4 #Knots
        self.aug1arg2 = 0.5 #Sigma
        self.aug1arg3 = 1 #mu
        self.aug1arg4 = 0.1 #min_dist
        self.aug1arg5 = np.nan
        self.augmentation2 = 'Noise'
        self.aug2arg1 = 1
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class WindowWarp:
    def __init__(self):
        self.augmentation1 = 'windowWarp'
        self.aug1arg1 = 4
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class WindowWarpThenNoise:
    def __init__(self):
        self.augmentation1 = 'windowWarpThenNoise'
        self.aug1arg1 = 4
        self.aug1arg2 = np.nan
        self.aug1arg3 = np.nan
        self.aug1arg4 = np.nan
        self.aug1arg5 = np.nan
        self.augmentation2 = 'Noise'
        self.aug2arg1 = 1
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100



class All_augments_by_score:
    def __init__(self):
        self.augmentation1 = 'All_augments_by_score'
        self.aug1arg1 = 0 #weighting of genauigkeit vs generalisierung. 0 = general, 1 = genau
        self.aug1arg2 = 2 #Method of dataset compilation. 1 = winner takes all, 2 = linear for all > 0, 3 = quadratic for all > 0
        self.aug1arg3 = 1 #Factor, how large Training dtaset is in relation to real data
        self.aug1arg4 = 0.25 #min_score_threshold
        self.aug1arg5 = 0 #save_trainings_vectors 0 = False, 1 = True
        self.augmentation2 = 'None'
        self.aug2arg1 = np.nan
        self.aug2arg2 = np.nan
        self.aug2arg3 = np.nan
        self.aug2arg4 = np.nan
        self.aug2arg5 = np.nan
        self.percentage_used = 100

class NN_Normal:
    def __init__(self):
        self.model = 'NN_Normal'

class Retrain_NN_Normal:
    def __init__(self):
        self.model = 'Retrain_NN_Normal'

class NN_Cheap:
    def __init__(self):
        self.model = 'NN_Cheap'

class NN_Cheap_formula_input:
    def __init__(self):
        self.model = 'NN_Cheap_formula_input'

class RF_Normal:
    def __init__(self):
        self.model = 'RF_Normal'

class RF_Cheap:
    def __init__(self):
        self.model = 'RF_Cheap'


def generate_test_arrays(l, n, m_values): #kann eig. weg
    """
    Generates a tuple of l NumPy arrays.
    :param l: Number of arrays to generate.
    :param n: Common size for the first dimension of all arrays.
    :param m_values: Array of numbers specifying the second dimension for each array.
    :return: A tuple of l NumPy arrays.
    """
    arrays = []
    for i in range(l):
        m = m_values[i]
        array = np.arange(1, n * m + 1).reshape(n, m)
        arrays.append(array)
    return tuple(arrays)

def save_arrays_as_pkl(arrays,prodtype):
    prodtype
    """
    Saves each array in the tuple 'arrays' as a .pkl file.

    :param arrays: Tuple of NumPy arrays to save.
    :param directory: Directory where the .pkl files will be saved.
    :param base_filename: Base name for the files. Files will be named as 'base_filename_1.pkl', 'base_filename_2.pkl', etc.
    """
    directory = CURRENT_DIRECTORY +r"\\Datensaetze\temp_for_augment"
    base_filename = "tmp_part"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filenames = []

    for i, array in enumerate(arrays, start=1):
        filename = os.path.join(directory, f"{base_filename}_{prodtype}_{i}.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(array, file)
        filenames.append(f"{base_filename}_{prodtype}_{i}")
    return filenames

def read_file(file_path):
    '''can read in .pkl and .npz files and returns them as an np.array'''
    if file_path.endswith('.npz'):
        with np.load(file_path) as data:
        # If the .npz file contains multiple arrays, concatenate them.
        # Otherwise, just load the single array.
            if len(data.files) > 1:
                arrays = [data[key] for key in data.files]
                return np.concatenate(arrays, axis=0)
            else:
                return data[data.files[0]]
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as file:
            return np.array(pickle.load(file))
    else:
        raise ValueError("Invalid file format. Please provide an NPZ or PKL file.")

def plot_multiple_predictions(data): #KANN EIGENTLICH WEG
    """
    Plots multiple channels with translucent lines.
    Parameters:
        data (np.array): An array with shape (m, n) where m is the number of channels and n is the time series data points.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot the time series data
    num_channels, num_points = data.shape
    for i in range(num_channels):
        axs[0].plot(data[i], alpha=0.25, color = "blue")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Value")
    axs[0].set_title("Multiple Predictions")
    
    std_over_time = np.std(data, axis=0)
    axs[1].plot(std_over_time)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Standard Deviation")
    axs[1].set_title("Standard Deviation Over Time")
    
    plt.tight_layout()
    plt.show()


def plot_multiple_channels(data: np.ndarray) -> None:
    m, n = data.shape  # m is number of channels, n is length of each channel
    fig, axs = plt.subplots(m, 1, figsize=(10, m*3))
    
    # Handling case where m=1 (array becomes 1D)
    if m == 1:
        axs.plot(data[0])
        sum_values = np.sum(data[0])
        axs.annotate(f'Sum = {sum_values}', xy=(0.8 * n, 0.8 * np.max(data[0])), xycoords='data')
        axs.set_title(f'Channel 0')
        axs.set_xlabel('Index')
        axs.set_ylabel('Value')
        return

    for i in range(m):
        axs[i].plot(data[i])
        sum_values = np.sum(data[i])
        axs[i].annotate(f'Sum = {sum_values}', xy=(0.8 * n, 0.8 * np.max(data[i])), xycoords='data')
        axs[i].set_title(f'Channel {i}')
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel('Value')
        
    plt.tight_layout()
    plt.show()

def create_single_ML_vector_optimized(channel_in: np.array) -> np.array:
    n = len(channel_in)
    channel_out = np.zeros((n - (PAST_VALUES + FUTURE_VALUES), PAST_VALUES + FUTURE_VALUES + 1))
    
    for i in range(n - (PAST_VALUES + FUTURE_VALUES)):
        channel_out[i, :PAST_VALUES] = channel_in[i:i + PAST_VALUES]
        channel_out[i, PAST_VALUES] = channel_in[i + PAST_VALUES + 1]
        channel_out[i, PAST_VALUES+1:] = channel_in[i + PAST_VALUES + 1: i + PAST_VALUES + FUTURE_VALUES + 1]
        
    return channel_out

def merge_ML_vectors_optimized(channels_in: tuple[np.array,...]) -> np.array:
    return np.concatenate(channels_in, axis=1)

def create_full_ml_vector_optimized(channels_in: tuple[np.array,...]) -> np.array:
    full_vector = create_single_ML_vector_optimized(channels_in[0])
    
    for i in range(1, len(channels_in)):
        vector_next_channel = create_single_ML_vector_optimized(channels_in[i])
        #full_vector = merge_ML_vectors_optimized((full_vector, vector_next_channel)) #--> TODO: If it still works, remove merge_ML_vectors_optimized
        full_vector = np.concatenate((full_vector, vector_next_channel), axis=1)
        
    return full_vector


def calc_va(training_channels_string, train_basis_data, augmented_train_basis_data, test_basis_data):
    train_data = []
    train_data_augmented = []
    test_data = []
    channelcounter = 0
    dt = 0.02
    if "v_x" in training_channels_string:
        if np.size(train_data) == 0:
            train_data = np.gradient(train_basis_data[channelcounter], dt)
            train_data_augmented = np.gradient(augmented_train_basis_data[channelcounter], dt)
            test_data = np.gradient(test_basis_data[channelcounter], dt)
            channelcounter += 1
        else:
            train_data = np.vstack((train_data,np.gradient(train_basis_data[channelcounter], dt)))
            train_data_augmented = np.vstack((train_data_augmented,np.gradient(augmented_train_basis_data[channelcounter], dt)))
            test_data = np.vstack((test_data,np.gradient(test_basis_data[channelcounter], dt)))
            channelcounter += 1
    #print(f"starting a_x with channelcounter {channelcounter}")
    if "a_x" in training_channels_string:
        if "v_x" in training_channels_string:
            channelcounter -= 1
        if np.size(train_data) == 0:
            train_data = np.gradient(np.gradient(train_basis_data[channelcounter], dt),dt)
            train_data_augmented = np.gradient(np.gradient(augmented_train_basis_data[channelcounter], dt),dt)
            test_data = np.gradient(test_basis_data[channelcounter], 1)
            channelcounter += 1
        else:
            #print(f"doing else a_x")
            train_data = np.vstack((train_data,np.gradient(np.gradient(train_basis_data[channelcounter], dt), dt)))
            train_data_augmented = np.vstack((train_data_augmented,np.gradient(np.gradient(augmented_train_basis_data[channelcounter], dt), dt)))
            test_data = np.vstack((test_data,np.gradient(np.gradient(test_basis_data[channelcounter], dt), dt)))
            channelcounter += 1
    #print(f"starting v_y with channelcounter {channelcounter}")
    if "v_y" in training_channels_string:
        if np.size(train_data) == 0:
            train_data = np.gradient(train_basis_data[channelcounter], dt)
            train_data_augmented = np.gradient(augmented_train_basis_data[channelcounter], dt)
            test_data = np.gradient(test_basis_data[channelcounter], dt)
            channelcounter += 1
        else:
            train_data = np.vstack((train_data,np.gradient(train_basis_data[channelcounter], dt)))
            train_data_augmented = np.vstack((train_data_augmented,np.gradient(augmented_train_basis_data[channelcounter], dt)))
            test_data = np.vstack((test_data,np.gradient(test_basis_data[channelcounter], dt)))
            channelcounter += 1
    #print(f"starting a_y with channelcounter {channelcounter}")
    if "a_y" in training_channels_string:
        if "v_y" in training_channels_string:
            channelcounter -= 1
        if np.size(train_data) == 0:
            train_data = np.gradient(np.gradient(train_basis_data[channelcounter], dt),dt)
            train_data_augmented = np.gradient(np.gradient(augmented_train_basis_data[channelcounter], dt),dt)
            test_data = np.gradient(test_basis_data[channelcounter], 1)
            channelcounter += 1
        else:
            #print(f"doing else a_x")
            train_data = np.vstack((train_data,np.gradient(np.gradient(train_basis_data[channelcounter], dt), dt)))
            train_data_augmented = np.vstack((train_data_augmented,np.gradient(np.gradient(augmented_train_basis_data[channelcounter], dt), dt)))
            test_data = np.vstack((test_data,np.gradient(np.gradient(test_basis_data[channelcounter], dt), dt)))
            channelcounter += 1
    if "v_z" in training_channels_string:
        if np.size(train_data) == 0:
            train_data = np.gradient(train_basis_data[channelcounter], dt)
            train_data_augmented = np.gradient(augmented_train_basis_data[channelcounter], dt)
            test_data = np.gradient(test_basis_data[channelcounter], dt)
            channelcounter += 1
        else:
            train_data = np.vstack((train_data,np.gradient(train_basis_data[channelcounter], dt)))
            train_data_augmented = np.vstack((train_data_augmented,np.gradient(augmented_train_basis_data[channelcounter], dt)))
            test_data = np.vstack((test_data,np.gradient(test_basis_data[channelcounter], dt)))
            channelcounter += 1
    if "a_z" in training_channels_string:
        if "v_z" in training_channels_string:
            channelcounter -= 1
        if np.size(train_data) == 0:
            train_data = np.gradient(np.gradient(train_basis_data[channelcounter], dt),dt)
            train_data_augmented = np.gradient(np.gradient(augmented_train_basis_data[channelcounter], dt),dt)
            test_data = np.gradient(test_basis_data[channelcounter], 1)
            channelcounter += 1
        else:
            #print(f"doing else a_x")
            train_data = np.vstack((train_data,np.gradient(np.gradient(train_basis_data[channelcounter], dt), dt)))
            train_data_augmented = np.vstack((train_data_augmented,np.gradient(np.gradient(augmented_train_basis_data[channelcounter], dt), dt)))
            test_data = np.vstack((test_data,np.gradient(np.gradient(test_basis_data[channelcounter], dt), dt)))
            channelcounter += 1
    if "v_sp" in training_channels_string:
        if np.size(train_data) == 0:
            train_data = np.gradient(train_basis_data[channelcounter], dt)
            train_data_augmented = np.gradient(augmented_train_basis_data[channelcounter], dt)
            test_data = np.gradient(test_basis_data[channelcounter], dt)
            channelcounter += 1
        else:
            train_data = np.vstack((train_data,np.gradient(train_basis_data[channelcounter], dt)))
            train_data_augmented = np.vstack((train_data_augmented,np.gradient(augmented_train_basis_data[channelcounter], dt)))
            test_data = np.vstack((test_data,np.gradient(test_basis_data[channelcounter], dt)))
            channelcounter += 1
    if "a_sp" in training_channels_string:
        if "v_sp" in training_channels_string:
            channelcounter -= 1
        if np.size(train_data) == 0:
            train_data = np.gradient(np.gradient(train_basis_data[channelcounter], dt),dt)
            train_data_augmented = np.gradient(np.gradient(augmented_train_basis_data[channelcounter], dt),dt)
            test_data = np.gradient(test_basis_data[channelcounter], 1)
            channelcounter += 1
        else:
            #print(f"doing else a_x")
            train_data = np.vstack((train_data,np.gradient(np.gradient(train_basis_data[channelcounter], dt), dt)))
            train_data_augmented = np.vstack((train_data_augmented,np.gradient(np.gradient(augmented_train_basis_data[channelcounter], dt), dt)))
            test_data = np.vstack((test_data,np.gradient(np.gradient(test_basis_data[channelcounter], dt), dt)))
            channelcounter += 1

#TODO: Remove if still works
####def remove_short_blocks_from_array(arr, block_length=6): 
####    '''TODO: Description'''
####    # Initialize variables to keep track of the start and end of blocks of ones
####    start = None
####    end = None
####
####    # Iterate through the array
####    for i in range(len(arr)):
####        # If current element is 1 and start is None, set the start
####        if arr[i] == 1 and start is None:
####            start = i
####
####        # If current element is 0 and start is not None, set the end
####        elif arr[i] == 0 and start is not None:
####            end = i
####            # If block length is less than the specified block_length, set the entire block to zeros
####            if end - start < block_length:
####                arr[start:end] = 0
####            start = None
####
####        # Handle the case for the last block in the array
####        elif i == len(arr) - 1 and start is not None:
####            end = i + 1
####            if end - start < block_length:
####                arr[start:end] = 0
####
####    return arr


def generate_descriptor_strings(data_params, aug_params,ML_params,additional_descriptor,augment_before_va):
    trainig_datapaths = data_params.trainig_datapaths
    validation_datapaths = data_params.validation_datapaths
    ####machine = data_params.machine
    ####training_channels = data_params.training_channels
    target_channels = data_params.target_channels
    data_name = data_params.name
    ####smoothing = data_params.smoothing
    ####modulo_split = data_params.modulo_split
    augmentation1 = aug_params.augmentation1
    aug1arg1 = aug_params.aug1arg1
    aug1arg2 = aug_params.aug1arg2
    aug1arg3 = aug_params.aug1arg3
    aug1arg4 = aug_params.aug1arg4
    aug1arg5 = aug_params.aug1arg5
    augmentation2 = aug_params.augmentation2
    aug2arg1 = aug_params.aug2arg1
    aug2arg2 = aug_params.aug2arg2
    aug2arg3 = aug_params.aug2arg3
    aug2arg4 = aug_params.aug2arg4
    aug2arg5 = aug_params.aug2arg5
    model = ML_params.model
    percentage_original_used = str(round(data_params.percentage_used,2)).replace('.', 'd')
    percentage_augmented_used = str(round(aug_params.percentage_used,2)).replace('.', 'd')

    #print(f"gendesstring: Augmentation1: {aug_params_test.augmentation1} aug1arg1: {aug_params_test.aug1arg1} aug1arg2: {aug_params_test.aug1arg2} aug1arg3: {aug_params_test.aug1arg3} aug1arg4: {aug_params_test.aug1arg4} aug1arg5: {aug_params_test.aug1arg5}")

    trainig_datapaths_string = "_".join(trainig_datapaths)
    validation_datapaths_string = "_".join(validation_datapaths)
    #training_channels_string = "_".join(training_channels)
    target_channels_string = "_".join(target_channels)
    descriptor_string_data = str("train_"+trainig_datapaths_string + "_valid_"+validation_datapaths_string+"_perc_orig_"+percentage_original_used+"_perc_aug_"+percentage_augmented_used)
    if True: #dont print args if nan
        if np.isnan(aug1arg1):
            aug1arg1_string = ""
        else:
            aug1arg1_string = "_"+str(aug1arg1).replace('.', 'd')
        if np.isnan(aug1arg2):
            aug1arg2_string = ""
        else:
            aug1arg2_string = "_"+str(aug1arg2).replace('.', 'd')
        if np.isnan(aug1arg3):
            aug1arg3_string = ""
        else:
            aug1arg3_string = "_"+str(aug1arg3).replace('.', 'd')
        if np.isnan(aug1arg4):
            aug1arg4_string = ""
        else:
            aug1arg4_string = "_"+str(aug1arg4).replace('.', 'd')
        if np.isnan(aug1arg5):
            aug1arg5_string = ""
        else:
            aug1arg5_string = "_"+str(aug1arg5).replace('.', 'd')
        if np.isnan(aug2arg1):
            aug2arg1_string = ""
        else:
            aug2arg1_string = "_"+str(aug2arg1).replace('.', 'd')
        if np.isnan(aug2arg2):
            aug2arg2_string = ""
        else:
            aug2arg2_string = "_"+str(aug2arg2).replace('.', 'd')
        if np.isnan(aug2arg3):
            aug2arg3_string = ""
        else:
            aug2arg3_string = "_"+str(aug2arg3).replace('.', 'd')
        if np.isnan(aug2arg4):
            aug2arg4_string = ""
        else:
            aug2arg4_string = "_"+str(aug2arg4).replace('.', 'd')
        if np.isnan(aug2arg5):
            aug2arg5_string = ""
        else:
            aug2arg5_string = "_"+str(aug2arg5).replace('.', 'd')

    #GENERATING DESCRIPTIVE STRINGS
    target_channels_aug = str("aug1_"+augmentation1+aug1arg1_string+aug1arg2_string+aug1arg3_string+aug1arg4_string+aug1arg5_string+"_"+
                              "aug2_"+augmentation2+aug2arg1_string+aug2arg2_string+aug2arg3_string+aug2arg4_string+aug2arg5_string)
    if augment_before_va == True:
        augment_when = "_augbeforeva"
    else:
        augment_when = "_augafterva"
    
    descriptor_string = str(data_name+"_"+model+"_"+target_channels_string+"_"+additional_descriptor+target_channels_aug+augment_when+"_"+descriptor_string_data)
    return trainig_datapaths_string, validation_datapaths_string, "None", target_channels_string, descriptor_string_data, descriptor_string

def calculate_derivatives(indata: np.ndarray) -> np.ndarray:
    if indata.shape[0] == 9:
        data = indata[:4]
        process_part = indata[4:]
    else:
        data = indata
    first_derivatives = np.gradient(data,DT, axis=1)
    second_derivatives = np.gradient(first_derivatives,DT, axis=1)
    
    if indata.shape[0] == 9:
        result =np.vstack((first_derivatives, second_derivatives,process_part))
    else:
        result =np.vstack((first_derivatives, second_derivatives))
    return result

def read_fulldata(name):
    #Format Aircut: pos_x, pos_y, pos_z, pos_sp, curr_x, curr_y, curr_z, curr_sp
    #Format process: pos_x, pos_y, pos_z, pos_sp, f_x, f_y, f_z, f_sp, materialremoved, curr_x, curr_y, curr_z, curr_sp
    #file_path = CURRENT_DIRECTORY + rf"\Datensaetze\preprocessed\{name}_alldata_allcurrent.pkl"
    file_path = os.path.join(CURRENT_DIRECTORY, "Datensaetze", "preprocessed", f"{name}_alldata_allcurrent.pkl")
    #file_path_process = CURRENT_DIRECTORY + rf"\Datensaetze\preprocessed\{name}_alldata_allforces_MRR_allcurrent.pkl"
    file_path_process = os.path.join(CURRENT_DIRECTORY, "Datensaetze", "preprocessed", f"{name}_alldata_allforces_MRR_allcurrent.pkl")
    #file_path_tfa = CURRENT_DIRECTORY + rf"\Datensaetze\temp_for_augment\{name}.pkl"
    file_path_tfa = os.path.join(CURRENT_DIRECTORY, "Datensaetze", "temp_for_augment", f"{name}.pkl")
    if os.path.exists(file_path):
        fulldata = read_file(file_path)
    elif os.path.exists(file_path_process):
        fulldata = read_file(file_path_process)
    elif os.path.exists(file_path_tfa):
        fulldata = read_file(file_path_tfa)
    else:
        print(f"ERROR: No file for {name} was found in {file_path} or {file_path_process}")

    #fulldata = fulldata[:, ::10] #downsampling by 10
    #print(f"readfulldata retunr ")
    return fulldata

def moving_average(array, window_size=41):
    if window_size == 1:
        return array
    # Get the number of rows
    num_rows = array.shape[0]
    
    # Initialize an empty array to store the smoothed data
    smoothed_array = np.zeros_like(array)
    
    # Create the moving average kernel
    kernel = np.ones(window_size) / window_size
    
    # Loop through each row and apply moving average
    for i in range(num_rows):
        smoothed_array[i, :] = np.convolve(array[i, :], kernel, mode='same')
        
    return smoothed_array

def get_target_current(target_channels_string, current_data):
    if target_channels_string == 'cur_x':
        train_target = current_data[0]
    elif target_channels_string == 'cur_y':
        train_target = current_data[1]
    elif target_channels_string == 'cur_z':
        train_target = current_data[2]
    elif target_channels_string == 'cur_sp':
        train_target = current_data[3]
    return train_target


def do_the_training(model, ML_params, full_train_data_ML_vector,val_data_ML_vector, test_data_ML_vector, full_train_target,val_target,test_target,model_string): #ONE OF THE MAIN FUNCTIONS
    if model == 'NN_Normal':
        predictions = trf.train_NN_Normal(model_string,ML_params, X_train = full_train_data_ML_vector, X_val =val_data_ML_vector,  X_test = test_data_ML_vector, y_train = full_train_target, y_val = val_target, y_test = test_target)
    if model == 'NN_Cheap':
        predictions = trf.train_NN_Cheap(model_string,ML_params, X_train = full_train_data_ML_vector, X_val =val_data_ML_vector,  X_test = test_data_ML_vector, y_train = full_train_target, y_val = val_target, y_test = test_target)
    if model == 'RF_Normal':
        predictions = trf.train_RF_Normal(model_string,ML_params, X_train = full_train_data_ML_vector, X_test = test_data_ML_vector, y_train = full_train_target, y_test = test_target)
    if model == 'RF_Cheap':
        predictions = trf.train_RF_Cheap(model_string,ML_params, X_train = full_train_data_ML_vector, X_test = test_data_ML_vector, y_train = full_train_target, y_test = test_target)

    if model == 'Retrain_NN_Normal':
        predictions = trf.continue_training_NN(model_string,ML_params, X_train = full_train_data_ML_vector, X_test = test_data_ML_vector, y_train = full_train_target, y_test = test_target)
    
    if model == 'NN_Cheap_formula_input':
        predictions = trf.train_NN_Cheap_formula_input(model_string,ML_params, X_train = full_train_data_ML_vector, X_test = test_data_ML_vector, y_train = full_train_target, y_test = test_target)
    return predictions

def do_the_augmentations_1(trainig_datapath, target_channels_string, augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4,aug1arg5, augmented_train_basis_data, train_target, train_target_augmented,model): #ONE OF THE MAIN FUNCTIONS

    if augmentation1  == 'Noise':
        ####print(f"running augmentation noise")
        augmented_train_basis_data = aug.add_noise(augmented_train_basis_data, aug1arg1)

    if augmentation1 == "AdaptiveNoise":
        noisefactor = aug1arg1
        window_length = 100# aug1arg2
        m, n = augmented_train_basis_data.shape
        channel_stdev = np.zeros(m)

        for i in range(m):
            channel_data = augmented_train_basis_data[i]
            window_stdevs = []

            for j in range(0, n-window_length+1):
                window_data = channel_data[j:j+window_length]
                stdev = np.std(window_data)
                window_stdevs.append(stdev)

            channel_stdev[i] = np.median(window_stdevs) #np.percentile(window_stdevs, 10)# np.median(window_stdevs)
        print(f"adaptive noises: {channel_stdev}")
        noise = np.random.normal(0, channel_stdev[i] * noisefactor, n)
        augmented_train_basis_data[i] += noise

    if augmentation1 == "AdaptiveNoiseSavgol":

        filter_length = 21#aug1arg2
        polyorder = 3#aug1arg3
        window_size = 10
        m, n = augmented_train_basis_data.shape  # m: number of channels, n: length of data
        std_devs = np.zeros(m)  # Array to store standard deviations of noise

        for i in range(m):
            channel_data = augmented_train_basis_data[i, :]
            # Apply Savitzky-Golay filter
            filtered_data = np.convolve(channel_data, np.ones(window_size)/window_size, mode='valid')
            noise = channel_data[:filtered_data.size] - filtered_data
            # Calculate and store standard deviation of noise
            std_devs[i] = np.std(noise)
        print(f"stddevs {std_devs}")
        a = 1/0
        
    if augmentation1  == 'homogenous_scaling':
        augmented_train_basis_data = aug.homogenous_scaling(augmented_train_basis_data, aug1arg1)
#TODO: In the literature, this is done to many time serieses that compose the training data. Here it is just done once --> Run it over the data several times
    if augmentation1 == 'windowWarpElong':
        len_train_data = augmented_train_basis_data.shape[0]
        len_train_target = train_target_augmented.shape[0]
        #train_stacked = np.vstack((augmented_train_basis_data, train_target))
        print(augmented_train_basis_data)
        train_stacked_aug = aug.window_warp(augmented_train_basis_data[0], 'elongate', warps = aug1arg1)#aug.window_warp(train_stacked, 'elongate', warps = aug1arg1) #
        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :]

    if augmentation1 == 'windowWarpShort':
        len_train_data = augmented_train_basis_data.shape[0]
        len_train_target = train_target_augmented.shape[0]
        train_stacked = np.vstack((augmented_train_basis_data, train_target))
        train_stacked_aug = aug.window_warp(train_stacked, 'shorten', warps = aug1arg1)
        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :]

    if augmentation1 == 'windowWarp':
        len_train_data = augmented_train_basis_data.shape[0]
        len_train_target = train_target_augmented.shape[0]
        train_stacked = np.vstack((augmented_train_basis_data, train_target))
        train_stacked_aug = aug.window_warp(train_stacked, aug1arg1)
        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :]

    if augmentation1 == 'magnitudeWarp':
        len_train_data = augmented_train_basis_data.shape[0]
        #len_train_target = train_target_augmented.shape[0]
        train_stacked = np.vstack((augmented_train_basis_data, train_target_augmented))
        train_stacked_aug = aug.magnitude_warping(train_stacked, knots = aug1arg1, sigma = aug1arg2, mu = aug1arg3, min_dist = aug1arg4)
        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :]

    if augmentation1 == 'rotation': #macht keinen sinn
        len_train_data = augmented_train_basis_data.shape[0]
        train_stacked = np.vstack((augmented_train_basis_data, train_target))
        train_stacked_aug = aug.rotation(train_stacked)
        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :]

    if augmentation1 == 'RandomDelete':
        keep_percentage = aug1arg1
        augmented_train_basis_data, train_target_augmented = sample_augmented_arrays(augmented_train_basis_data, train_target_augmented, keep_percentage)

    if augmentation1 == 'timeWarp':
        len_train_data = augmented_train_basis_data.shape[0]
        train_stacked = np.vstack((augmented_train_basis_data, train_target_augmented))
        train_stacked_aug = aug.time_warping(train_stacked, knots = aug1arg1, sigma = aug1arg2, mu = aug1arg3, min_dist = aug1arg4)
        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :]

    if augmentation1 == 'TimeVAEgenerated': #THIS ONLY WORKS FOR THE SPECIAL CASE OF TRAINING ALU AIRCUT
        len_train_data = augmented_train_basis_data.shape[0]
        folder_path = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\generated_TimeVAE"
        read_data = np.array([])
        for file_name in os.listdir(folder_path):
            #print(f"chekcing {file_name} for string {trainig_datapath[:-2]}")
            if trainig_datapath[:-2] in file_name and file_name.endswith('.pkl'):
                #print("TimeVAEgenerated found it")
                with open(os.path.join(folder_path, file_name), 'rb') as f:
                    read_data = pickle.load(f)
        if np.array_equal(read_data, np.array([])):
            print(f"\n\nFATAL ERROR: {trainig_datapath[:-2]} not found\n\n")

        this_sample = int(random.uniform(0, read_data.shape[0]))
        #print(f"read_data {read_data.shape}")
        this_read_data = read_data[this_sample,:,:].T

        #print(f"read_data {this_read_data.shape}")
        #train_stacked_aug = aug.TimeVAEgenerated(aug1arg1)
        augmented_train_basis_data = this_read_data[:len_train_data, :]
        if target_channels_string == "cur_x":
            train_target_augmented = this_read_data[-4, :].reshape(1, -1)
        if target_channels_string == "cur_y":
            train_target_augmented = this_read_data[-3, :].reshape(1, -1)
        if target_channels_string == "cur_z":
            train_target_augmented = this_read_data[-2, :].reshape(1, -1) 
        if target_channels_string == "cur_sp":
            train_target_augmented = this_read_data[-1, :].reshape(1, -1)

    #if augmentation1 = 'TimeVAEgenerated_stacked':
    
    
    if augmentation1 == 'ydata_augmentation':
        len_train_data = augmented_train_basis_data.shape[0]
        train_stacked_aug = aug.YData_generated()
        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :] 

    if augmentation1 == 'MSE_threshold':
        threshold = aug1arg1
        mse_folder = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\MSE_values"
        ####print(f"MSE_treshold augmented_train_basis_data {augmented_train_basis_data.shape}")
        ####print(f"MSE_treshold train_target_augmented {train_target_augmented.shape}")
        this_MSEs = []

        for filename in os.listdir(mse_folder):
            if trainig_datapath in filename and target_channels_string in filename:
                file_path = os.path.join(mse_folder, filename)
                with open(file_path, 'rb') as f:
                    mse_values = pickle.load(f)
                this_MSEs.extend(mse_values)
        this_MSEs = np.array(this_MSEs)
        #print(f"thismses pre pad {this_MSEs.shape}")
        #padlen = 4#CUTOFF #TBH i dont really know why this needs a 4 isntead of the CUTOFF. Most likely has to do with the fact, that the files in mse_folder were trained with 4
        this_MSEs = np.pad(this_MSEs, (0, CUTOFF), 'constant', constant_values=(0))
        #print(f"thismses after pad {this_MSEs.shape}")
        ####print(this_MSEs.shape)
        norm_this_MSEs = this_MSEs / np.mean(this_MSEs)
        len_train_data = augmented_train_basis_data.shape[0]
        train_stacked = np.vstack((augmented_train_basis_data, train_target_augmented))
        ###Do augmentation here
        mask = norm_this_MSEs >= threshold
        train_stacked_aug = train_stacked[:, mask]


        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :]
        ####print(f"after MSE_treshold augmented_train_basis_data {augmented_train_basis_data.shape}")
        ####print(f"after MSE_treshold train_target_augmented {train_target_augmented.shape}")
    
    if augmentation1 == 'All_augments_by_score': #a bit hacky
        print(f"doing All_augments_by_score")
        min_score_threshold = aug1arg3 #0.25
        ALL_AUGMENTS = {
        "MSE_threshold": MSE_threshold(),
        "MW": MagnitudeWarp(),
        "MWthenTW": MagnitudeWarpThenTimeWarp(),
        "Noise": Noise(),
        "RD": RandomDelete(),
        "TW": TimeWarp(),
        "TWthenNoise": TimeWarpThenNoise(),
        "WW": WindowWarp(),
        "WWthenNoise": WindowWarpThenNoise(),
        }
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        if "Air" in trainig_datapath:
            for char in ['x', 'y', 'z']:
                if char in target_channels_string:
                    df_accuracy = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_accuracy_air_{char}.csv")
                    df_generalisation = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_generalisation_air_{char}.csv")
            #if any(char in target_channels_string for char in 'xyz'):
            #    df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_air_xyz.csv")
            #    df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_air_xyz.csv")
            if "sp" in target_channels_string:
                df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_air_sp.csv")
                df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_air_sp.csv")
        if "Mat" in trainig_datapath:
            for char in ['x', 'y', 'z']:
                if char in target_channels_string:
                    df_accuracy = pd.read_csv(current_directory + rf"\\Auswertung\\bewertung_der_methoden\\score_accuracy_mat_{char}.csv")
                    df_generalisation = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_generalisation_mat_{char}.csv")
            #if any(char in target_channels_string for char in 'xyz'):
            #    df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_mat_xyz.csv")
            #    df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_mat_xyz.csv")
            if "sp" in target_channels_string:
                df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_mat_sp.csv")
                df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_mat_sp.csv")

        focus = aug1arg1 #self.aug1arg1 = 0.5 #weighting of genauigkeit vs generalisierung. 0 = general, 1 = genau
        composition_method = aug1arg2
        list_full_train_data = []
        list_full_train_target = []

        total_score = 0
        highest_score = 0
        for augment in ALL_AUGMENTS:
            #this_augmentation = ALL_AUGMENTS[augment]
            score_accuracy = df_accuracy.query(f"`Method` == '{augment}'")[model].iloc[0]
            score_generalisation = df_generalisation.query(f"`Method` == '{augment}'")[model].iloc[0]
            score_this_focus = focus * score_accuracy + (1-focus) * score_generalisation
            if score_this_focus <= min_score_threshold: continue
            if composition_method == 3:
                score_this_focus = score_this_focus * score_this_focus
            if True:#score_this_focus > MIN_SCORE_THRESHOLD:
                total_score += score_this_focus
                print(f"adding for {augment} {score_this_focus} to a total of {total_score}")
                if highest_score < score_this_focus:
                    highest_score = score_this_focus

        for augment in ALL_AUGMENTS:
            this_augmentation = ALL_AUGMENTS[augment]
            score_accuracy = df_accuracy.query(f"`Method` == '{augment}'")[model].iloc[0]
            score_generalisation = df_generalisation.query(f"`Method` == '{augment}'")[model].iloc[0]
            score_this_focus = focus * score_accuracy + (1-focus) * score_generalisation
            if score_this_focus <= min_score_threshold: continue
            if composition_method == 3:
                score_this_focus = score_this_focus * score_this_focus
            print(f"{augment} has a score of {score_this_focus} of {total_score}")

            augmentation1 = this_augmentation.augmentation1
            aug1arg1 = this_augmentation.aug1arg1
            aug1arg2 = this_augmentation.aug1arg2
            aug1arg3 = this_augmentation.aug1arg3
            aug1arg4 = this_augmentation.aug1arg4
            aug1arg5 = this_augmentation.aug1arg5
            augmentation2 = this_augmentation.augmentation2
            aug2arg1 = this_augmentation.aug2arg1
            aug2arg2 = this_augmentation.aug2arg2
            aug2arg3 = this_augmentation.aug2arg3
            aug2arg4 = this_augmentation.aug2arg4
            aug2arg5 = this_augmentation.aug2arg5
            percentage_augment_used = this_augmentation.percentage_used

            if composition_method == 1:
                if score_this_focus != highest_score:
                    continue
                else:
                    noop = 1
                    #print(f"highest score has {augment} with {score_this_focus}")
            elif composition_method == 2 or composition_method == 3:
                percentage_augment_used = (score_this_focus/total_score)*100
                if percentage_augment_used < 0.1: continue #else, there would be the possibility to not return any data and break the script
                #print(f"using {percentage_augment_used}% of {augment}")

            this_augmented_train_basis_data, this_train_target_augmented = do_the_augmentations_1(trainig_datapath, target_channels_string, augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4,aug1arg5, augmented_train_basis_data, train_target_augmented, train_target_augmented,model)
            this_augmented_train_basis_data, this_train_target_augmented = do_the_augmentations_2(trainig_datapath, target_channels_string, augmentation2, aug2arg1, aug2arg2, aug2arg3, aug2arg4,aug2arg5, augmented_train_basis_data, train_target_augmented,model)
            #print(f"after augment {augment}: augmented_train_basis_data: {this_augmented_train_basis_data.shape} train_target_augmented {this_train_target_augmented.shape}")
           
            list_full_train_data.append(this_augmented_train_basis_data)
            list_full_train_target.append(this_train_target_augmented)
            #if augment != "MSE_threshold":
            #    DEBUGaugmented_train_basis_data = np.vstack(list_full_train_data)
            #    DEBUGtrain_target_augmented = np.concatenate(list_full_train_target).reshape(1, -1)
            #    print(f"after augment {augment}: DEBUGaugmented_train_basis_data: {DEBUGaugmented_train_basis_data.shape} DEBUGtrain_target_augmented {DEBUGtrain_target_augmented.shape}")
        if total_score > 0:
            augmented_train_basis_data = np.concatenate(list_full_train_data, axis=1)
            train_target_augmented = np.concatenate(list_full_train_target,axis=1)#.reshape(1, -1)
        #print(f"All_augments_by_score returning augmented_train_basis_data {augmented_train_basis_data.shape} train_target_augmented {train_target_augmented.shape}")
            
    
    #print(f"aug1 returning augmented_train_basis_data {augmented_train_basis_data.shape} train_target_augmented {train_target_augmented.shape}")
    #ugmented_train_basis_data (4, 25202) train_target_augmented (1, 25202)
    return augmented_train_basis_data, train_target_augmented
    
def do_the_augmentations_2(trainig_datapath, target_channels_string, augmentation2, aug2arg1, aug2arg2, aug2arg3, aug2arg4,aug2arg5, augmented_train_basis_data, train_target_augmented,model): #CURRENTLY NOT ALL IMPLEMENTED
    if augmentation2 == 'timeWarp':
        #print(f"doing timeWarp 2")
        len_train_data = augmented_train_basis_data.shape[0]
        train_stacked = np.vstack((augmented_train_basis_data, train_target_augmented))
        train_stacked_aug = aug.time_warping(train_stacked, knots = aug2arg1, sigma = aug2arg2, mu = aug2arg3, min_dist = aug2arg4)
        augmented_train_basis_data = train_stacked_aug[:len_train_data, :]
        train_target_augmented = train_stacked_aug[len_train_data:, :]

    if augmentation2  == 'Noise':
        ####print(f"running augmentation noise")
        augmented_train_basis_data = aug.add_noise(augmented_train_basis_data, aug2arg1)

    #print(f"doaugment pre sample returning augmented_train_basis_data {augmented_train_basis_data.shape} train_target_augmented {train_target_augmented.shape}")
    #augmented_train_basis_data, train_target_augmented = sample_augmented_arrays(augmented_train_basis_data, train_target_augmented, percentage_augment_used) 
    #print(f"doaugment2 post sample augmented_train_basis_data {augmented_train_basis_data.shape} train_target_augmented {train_target_augmented.shape}")
    return augmented_train_basis_data, train_target_augmented

def do_augmentations_after_vectorization(augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4,aug1arg5,trainig_datapaths, train_data_ML_vector, train_target_augmented, train_target):

    if augmentation1 == 'Overtrain_bad_MSEs':
        path_bad_MSEs = CURRENT_DIRECTORY + fr"\Auswertung\overtrain_bad_predictions\bad_predictions_{trainig_datapaths[i]}_MSEvalues_aug1arg3_{aug1arg3}.pkl"
        save_MSE_augmented_path = CURRENT_DIRECTORY + fr"\Datensaetze\bad_MSE_merged\bad_predictions_merged_vectors_{trainig_datapaths[i]}_aug1arg1_{aug1arg1}_aug1arg2_{aug1arg2}_aug1arg3_{aug1arg3}_data.pkl"
        save_MSE_augmented_target_path = CURRENT_DIRECTORY+ fr"C\Datensaetze\bad_MSE_merged\bad_predictions_merged_vectors_{trainig_datapaths[i]}_aug1arg1_{aug1arg1}_aug1arg2_{aug1arg2}_aug1arg3_{aug1arg3}_target.pkl"
        if os.path.exists(save_MSE_augmented_path):
            modified_train_data_ML_vector = read_file(save_MSE_augmented_path)
            train_target_augmented = read_file(save_MSE_augmented_target_path)
        else:
            MSE_array = read_file(path_bad_MSEs)
            modified_train_data_ML_vector = train_data_ML_vector[0]
            train_target_augmented = train_target[:,:1]
            avg_MSE = np.mean(MSE_array)#np.full_like(MSE_array, np.mean(MSE_array))
            #print(f"average MSE = {avg_MSE}")
            if aug1arg1 == 1:
                MSE_array = np.sqrt(MSE_array/avg_MSE)
            if aug1arg1 == 2: #0.00607
                MSE_array = (MSE_array/avg_MSE)/5
            if aug1arg1 == 3:
                MSE_array = (MSE_array/avg_MSE)/2
            if aug1arg1 == 4:
                MSE_array = (MSE_array/avg_MSE)
            if aug1arg1 == 5:
                new_arr = np.zeros_like(MSE_array)
                new_arr[MSE_array > (avg_MSE*aug1arg2)] = 1
                MSE_array = new_arr
            if aug1arg1 == 6:
                new_arr = np.zeros_like(MSE_array)
                new_arr[MSE_array > (avg_MSE*aug1arg2)] = 1
                MSE_array = np.logical_not(new_arr).astype(float)
            if aug1arg1 == 7:
                MSE_array = (MSE_array/avg_MSE)/aug1arg2
            integral = 0
            for i in range(len(MSE_array)):
                integral += (MSE_array[i])
                while integral > 1:
                    if integral > 1:
                        modified_train_data_ML_vector = np.vstack((modified_train_data_ML_vector, train_data_ML_vector[i]))
                        train_target_augmented = np.hstack((train_target_augmented, train_target[:, i:i+1]))
                        integral -= 1
            print(f"Finished generation of modified_train_data_ML_vector: {modified_train_data_ML_vector.shape} and target: {train_target_augmented.shape}")
            with open(save_MSE_augmented_path, 'wb') as file:
                pickle.dump(modified_train_data_ML_vector, file)
            with open(save_MSE_augmented_target_path, 'wb') as file:
                pickle.dump(train_target_augmented, file)
        train_data_ML_vector_augmented = modified_train_data_ML_vector
    
    if augmentation1 == 'smogn_augmentation': #Takes about 20 minutes #THIS SHOULD BE PART OF do_augmentations_after_vectorization()
        print(f"this would run smogn_augmentation train vector: {full_train_data_ML_vector.shape} target {full_train_target.shape}\n\n")
        full_train_data_ML_vector = full_train_data_ML_vector.T
        # i need it to be like (4, 46093) target (1, 46093)


        #a = 1/0
        len_train_data = full_train_data_ML_vector.shape[0]
        len_train_target = full_train_target.shape[0]
        train_stacked = np.vstack((full_train_data_ML_vector, full_train_target))
        this_df = pd.DataFrame(train_stacked.T, columns=[f'Column_{i+1}' for i in range(len_train_data+len_train_target)])
        y = str(f"Column_{len_train_data+len_train_target}")
        train_stacked_aug_df = aug.smogn_augemnt(this_df, y) #the 'current' part does not really matter tbh, since we always only use one channels as target, the last one in the array
        save_smogn_generated_path = CURRENT_DIRECTORY + r"\Datensaetze\generated_smogn\CMX_Alu_Tr_Air_smogn_generated.pkl"
        full_train_data_ML_vector = train_stacked_aug_df.to_numpy().T
        with open(save_smogn_generated_path, 'wb') as file:
            pickle.dump(full_train_data_ML_vector, file)
        print(f"\nfinished smoting: {full_train_data_ML_vector.shape}")
        full_train_target  =full_train_data_ML_vector[-1:, :]
        full_train_data_ML_vector = full_train_data_ML_vector[:-1, :].T
        print(f"\nfinished smogn train vector: {full_train_data_ML_vector.shape} target {full_train_target.shape}\n")
    
    if augmentation1 == "LimitTrainingdata":  #TODO: Auch für Augemntation2
        ####print("LimitTrainingdata")
        ####print(train_data_ML_vector.shape)
        ####print(train_target_augmented.shape)
        
        merged_array = np.hstack((train_data_ML_vector, train_target_augmented.T))
        remove_chance = 100-aug1arg1
        random_values = np.random.uniform(0, 100, merged_array.shape[0])
        mask = random_values >= remove_chance
        # Remove elements from all channels
        filtered_arr = merged_array[mask, :]



        train_data_ML_vector, train_target_augmented = np.hsplit(filtered_arr, [train_data_ML_vector.shape[1]])
        train_target_augmented = train_target_augmented.T
        ####print(f"after shortening")
        ####print(train_data_ML_vector.shape)
        ####print(train_target_augmented.shape)

    return train_data_ML_vector, train_target_augmented
    noop = 1

def plot_two_channels(arr, step = 1):
    x_axis = np.arange(0, arr.shape[0], step)
    channel1 = arr[::step, 0]
    channel2 = arr[::step, 1]
    
    fig, axes = plt.subplots(2, 1, sharex=True)
    
    axes[0].scatter(x_axis, channel1, alpha=0.5, s=10)
    axes[0].set_title('Channel 1')
    axes[0].set_ylabel('Value')
    
    axes[1].scatter(x_axis, channel2, alpha=0.5, s=10)
    axes[1].set_title('Channel 2')
    axes[1].set_xlabel('Index or Time')
    axes[1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()

def calculate_moving_mse(mean_predictions, mean_actual_values, window_length=21):# This could be its own module in a .py file
    half_window = window_length // 2
    this_MSEs = []
    
    for i in range(len(mean_predictions)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(mean_predictions), i + half_window + 1)
        
        window_preds = mean_predictions[start_idx:end_idx]
        window_actual = mean_actual_values[start_idx:end_idx]
        
        mse = mean_squared_error(window_actual, window_preds)
        this_MSEs.append(mse)
    
    return np.array(this_MSEs)

def sample_arrays(arrayA, arrayB, percentage_used):
    if percentage_used == 100:
        return arrayA, arrayB
    assert arrayA.shape[0] == arrayB.shape[1], "The number of data points in both arrays should be the same."
    
    total_data_points = arrayA.shape[0]
    random_flags = np.random.rand(total_data_points) < (percentage_used / 100.0)
    
    sampled_arrayA = arrayA[random_flags, :]
    sampled_arrayB = arrayB[:, random_flags]
    return sampled_arrayA, sampled_arrayB

def sample_augmented_arrays(arrayA, arrayB, percentage_used):
    
    if percentage_used == 100:
        return arrayA, arrayB

    total_data_points = arrayA.shape[1]
    random_flags = np.random.rand(total_data_points) < (percentage_used / 100.0)
    
    sampled_arrayA = arrayA[:, random_flags]
    sampled_arrayB = arrayB[:, random_flags]
    return sampled_arrayA, sampled_arrayB

def sample_ML_arrays(arrayA, arrayB, percentage_used):
    
    if percentage_used == 100:
        return arrayA, arrayB

    total_data_points = arrayA.shape[0]
    random_flags = np.random.rand(total_data_points) < (percentage_used / 100.0)
    
    sampled_arrayA = arrayA[random_flags, :]
    sampled_arrayB = arrayB[:, random_flags]
    return sampled_arrayA, sampled_arrayB

def augment_arrays_for_max(arrays,trainig_datapath,target_channels_string,model):
    #arrays = tuple of arrays shape(channels, datapoints) taht shall be augmented
    #trainig_datapath = string that contains "Air" or "Mat", depending on if aircut or not
    #target_channels_string = string that contains x,y,z, or sp, depending on the axis
    #model = ML_params.model
    augmented_train_basis_data, train_target_augmented = 1#TODO change
    augment_before_va = True
    MIN_SCORE_THRESHOLD = 0.25
    ALL_AUGMENTS = {
    #"MSE_threshold": MSE_threshold(), #comment out since not available
    "MW": MagnitudeWarp(),
    "MWthenTW": MagnitudeWarpThenTimeWarp(),
    "Noise": Noise(),
    "RD": RandomDelete(),
    "TW": TimeWarp(),
    "TWthenNoise": TimeWarpThenNoise(),
    "WW": WindowWarp(),
    "WWthenNoise": WindowWarpThenNoise(),
    }
    if "Air" in trainig_datapath:
        for char in ['x', 'y', 'z']:
            if char in target_channels_string:
                df_accuracy = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_accuracy_air_{char}.csv")
                df_generalisation = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_generalisation_air_{char}.csv")
        #if any(char in target_channels_string for char in 'xyz'):
        #    df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_air_xyz.csv")
        #    df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_air_xyz.csv")
        if "sp" in target_channels_string:
            df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_air_sp.csv")
            df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_air_sp.csv")
    if "Mat" in trainig_datapath:
        for char in ['x', 'y', 'z']:
            if char in target_channels_string:
                df_accuracy = pd.read_csv(current_directory + rf"\\Auswertung\\bewertung_der_methoden\\score_accuracy_mat_{char}.csv")
                df_generalisation = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_generalisation_mat_{char}.csv")
        #if any(char in target_channels_string for char in 'xyz'):
        #    df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_mat_xyz.csv")
        #    df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_mat_xyz.csv")
        if "sp" in target_channels_string:
            df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_mat_sp.csv")
            df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_mat_sp.csv")

    focus = 0 #self.aug1arg1 = 0.5 #weighting of genauigkeit vs generalisierung. 0 = general, 1 = genau
    composition_method = 2
    list_full_train_data = []
    list_full_train_target = []
    total_score = 0
    highest_score = 0
    for augment in ALL_AUGMENTS:
        #this_augmentation = ALL_AUGMENTS[augment]
        score_accuracy = df_accuracy.query(f"`Method` == '{augment}'")[model].iloc[0]
        score_generalisation = df_generalisation.query(f"`Method` == '{augment}'")[model].iloc[0]
        score_this_focus = focus * score_accuracy + (1-focus) * score_generalisation
        if score_this_focus <= MIN_SCORE_THRESHOLD: continue
        if composition_method == 3:
            score_this_focus = score_this_focus * score_this_focus
        if True:#score_this_focus > MIN_SCORE_THRESHOLD:
            total_score += score_this_focus
            #print(f"adding for {augment} {score_this_focus} to a total of {total_score}")
            if highest_score < score_this_focus:
                highest_score = score_this_focus
    
    for augment in ALL_AUGMENTS:
        this_augmentation = ALL_AUGMENTS[augment]
        score_accuracy = df_accuracy.query(f"`Method` == '{augment}'")[model].iloc[0]
        score_generalisation = df_generalisation.query(f"`Method` == '{augment}'")[model].iloc[0]
        score_this_focus = focus * score_accuracy + (1-focus) * score_generalisation
        if score_this_focus <= MIN_SCORE_THRESHOLD: continue
        if composition_method == 3: #doesnt happen, set to 2
            score_this_focus = score_this_focus * score_this_focus
        #print(f"{augment} has a score of {score_this_focus} of {total_score}")

        subaugmentation1 = this_augmentation.augmentation1
        subaug1arg1 = this_augmentation.aug1arg1
        subaug1arg2 = this_augmentation.aug1arg2
        subaug1arg3 = this_augmentation.aug1arg3
        subaug1arg4 = this_augmentation.aug1arg4
        subaug1arg5 = this_augmentation.aug1arg5
        subaugmentation2 = this_augmentation.augmentation2
        subaug2arg1 = this_augmentation.aug2arg1
        subaug2arg2 = this_augmentation.aug2arg2
        subaug2arg3 = this_augmentation.aug2arg3
        subaug2arg4 = this_augmentation.aug2arg4
        subaug2arg5 = this_augmentation.aug2arg5
        subpercentage_augment_used = this_augmentation.percentage_used

        if composition_method == 1: #doesnt happen, set to 2
            if score_this_focus != highest_score:
                continue
            else:
                noop = 1
                #print(f"highest score has {augment} with {score_this_focus}")
        elif composition_method == 2 or composition_method == 3:
            subpercentage_augment_used = (score_this_focus/total_score)*100
            if subpercentage_augment_used < 0.1: continue #else, there would be the possibility to not return any data and break the script
            #print(f"using {percentage_augment_used}% of {augment}")

        this_augmented_train_basis_data, this_train_target_augmented = do_the_augmentations_1(trainig_datapath, target_channels_string, subaugmentation1, subaug1arg1, subaug1arg2, subaug1arg3, subaug1arg4, subaug1arg5, augmented_train_basis_data, train_target_augmented, train_target_augmented,model)
        this_augmented_train_basis_data, this_train_target_augmented = do_the_augmentations_2(trainig_datapath, target_channels_string, subaugmentation2, subaug2arg1, subaug2arg2, subaug2arg3, subaug2arg4, subaug2arg5, augmented_train_basis_data, train_target_augmented,model)
        #print(f"after augment {augment}: augmented_train_basis_data: {this_augmented_train_basis_data.shape} train_target_augmented {this_train_target_augmented.shape}")
           
        if augment_before_va == True:
            this_train_data_augmented = calculate_derivatives(this_augmented_train_basis_data)

            
        this_train_data_ML_vector_augmented = create_full_ml_vector_optimized(this_train_data_augmented)
        this_train_target_augmented = this_train_target_augmented[:,:-CUTOFF]

        #DOING AUGMENTATION, THAT HAPPENS AFTER CREATION OF THE VECTORS
        #notneededformax
        #this_train_data_ML_vector_augmented, this_train_target_augmented = do_augmentations_after_vectorization(augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4,aug1arg5,trainig_datapaths, this_train_data_ML_vector_augmented, this_train_target_augmented, train_target)

        #SAMPLE VECTOR HERE
        #print(f"PRE subpercentage_augment_used {subpercentage_augment_used} this_train_data_ML_vector_augmented {this_train_data_ML_vector_augmented.shape} this_train_target_augmented {this_train_target_augmented.shape}")
        this_train_data_ML_vector_augmented, this_train_target_augmented = sample_ML_arrays(this_train_data_ML_vector_augmented, this_train_target_augmented, subpercentage_augment_used)
        #print(f"POST subpercentage_augment_used {subpercentage_augment_used} this_train_data_ML_vector_augmented {this_train_data_ML_vector_augmented.shape} this_train_target_augmented {this_train_target_augmented.shape}")


        list_full_train_data.append(this_train_data_ML_vector_augmented)
        list_full_train_target.append(this_train_target_augmented)


    return arrays

def transform_array(arr): #JUST FOR TESTING, MOST LIKELY SHIT
    if arr.shape[0] != 13:
        raise ValueError("The array does not have the correct shape (13, n).")
    
    value_shift = 0
    y_dot = arr[0+value_shift]  # Second value
    y_dotdot = arr[4+value_shift]  # Sixth value
    MMR = arr[-1]  # Last value
    
    # Calculate required variables
    sgn_y_dot = np.sign(y_dot)
    y_dot_mul_y_dotdot = y_dot * y_dotdot
    y_dot_squared = y_dot ** 2
    
    # Create new array with shape (4, n)
    new_array = np.vstack([sgn_y_dot, MMR, y_dot_mul_y_dotdot, y_dot_squared])
    
    return new_array

def do_get_train_data_and_targetvectors(trainig_datapath, trainig_datapaths,augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4,aug1arg5,augmentation2, aug2arg1, aug2arg2, aug2arg3, aug2arg4,aug2arg5,augment_before_va,target_channels_string,percentage_original_used,percentage_augmented_used,model):
    #Getting train basis data and target
    fulldata = read_fulldata(trainig_datapath)
    train_basis_data = fulldata[:-4,:]

    if FILL_PROCESS_FOR_AIRCUT == True: #fills in process forces (as zeros) for aircut so that it can be used with process data. else the vectors would have different sizes so it would not be possible to merge them
        if "Air" in trainig_datapath:
            padding = ((0, 5), (0, 0))
            train_basis_data = np.pad(train_basis_data, padding, 'constant', constant_values=0)



    train_basis_data = moving_average(train_basis_data, window_size=WINDOWSIZE)
    train_target = get_target_current(target_channels_string, fulldata[-4:,:]).reshape((1, -1))
    train_target = np.abs(train_target)
    if augment_before_va == False:
        train_basis_data = calculate_derivatives(train_basis_data)
    #AUGMENTING THE DATA IN THE PREVIOUSLY GENERATE ARRAYS
    augmented_train_basis_data = np.copy(train_basis_data)
    train_target_augmented = np.copy(train_target)
    if augmentation1 == "All_augments_by_score":
        if target_channels_string in "cur_x, cur_y, cur_z":
            percentage_original_used = 43
        elif target_channels_string == "cur_sp":
            percentage_augmented_used = 81
        train_data = calculate_derivatives(train_basis_data)
        train_data_ML_vector = create_full_ml_vector_optimized(train_data) #optimized means just, that i changed the function to make it faster. The result is the same
        train_target = train_target[:,:-CUTOFF] 

        MIN_SCORE_THRESHOLD = 0.25
        ALL_AUGMENTS = {
        #"MSE_threshold": MSE_threshold(), #comment out for Paula since she needs 12-5
        "MW": MagnitudeWarp(),
        "MWthenTW": MagnitudeWarpThenTimeWarp(),
        "Noise": Noise(),
        "RD": RandomDelete(),
        "TW": TimeWarp(),
        "TWthenNoise": TimeWarpThenNoise(),
        "WW": WindowWarp(),
        "WWthenNoise": WindowWarpThenNoise(),
        }
        
        current_directory = os.path.dirname(os.path.abspath(__file__))
        if "Air" in trainig_datapath:
            for char in ['x', 'y', 'z']:
                if char in target_channels_string:
                    df_accuracy = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_accuracy_air_{char}.csv")
                    df_generalisation = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_generalisation_air_{char}.csv")

            if "sp" in target_channels_string:
                df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_air_sp.csv")
                df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_air_sp.csv")
        if "Mat" in trainig_datapath:
            for char in ['x', 'y', 'z']:
                if char in target_channels_string:
                    df_accuracy = pd.read_csv(current_directory + rf"\\Auswertung\\bewertung_der_methoden\\score_accuracy_mat_{char}.csv")
                    df_generalisation = pd.read_csv(current_directory + fr"\\Auswertung\\bewertung_der_methoden\\score_generalisation_mat_{char}.csv")

            if "sp" in target_channels_string:
                df_accuracy = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_accuracy_mat_sp.csv")
                df_generalisation = pd.read_csv(current_directory+ r"\Auswertung\bewertung_der_methoden\score_generalisation_mat_sp.csv")

        focus = aug1arg1 #self.aug1arg1 = 0.5 #weighting of genauigkeit vs generalisierung. 0 = general, 1 = genau
        composition_method = aug1arg2
        list_full_train_data = []
        list_full_train_target = []

        total_score = 0
        highest_score = 0
        for augment in ALL_AUGMENTS:
            #this_augmentation = ALL_AUGMENTS[augment]
            score_accuracy = df_accuracy.query(f"`Method` == '{augment}'")[model].iloc[0]
            score_generalisation = df_generalisation.query(f"`Method` == '{augment}'")[model].iloc[0]
            score_this_focus = focus * score_accuracy + (1-focus) * score_generalisation
            if score_this_focus <= MIN_SCORE_THRESHOLD: continue
            if composition_method == 3:
                score_this_focus = score_this_focus * score_this_focus
            if True:#score_this_focus > MIN_SCORE_THRESHOLD:
                total_score += score_this_focus
                #print(f"adding for {augment} {score_this_focus} to a total of {total_score}")
                if highest_score < score_this_focus:
                    highest_score = score_this_focus

        for augment in ALL_AUGMENTS:
            this_augmentation = ALL_AUGMENTS[augment]
            score_accuracy = df_accuracy.query(f"`Method` == '{augment}'")[model].iloc[0]
            score_generalisation = df_generalisation.query(f"`Method` == '{augment}'")[model].iloc[0]
            score_this_focus = focus * score_accuracy + (1-focus) * score_generalisation
            if score_this_focus <= MIN_SCORE_THRESHOLD: continue
            if composition_method == 3:
                score_this_focus = score_this_focus * score_this_focus
            #print(f"{augment} has a score of {score_this_focus} of {total_score}")

            subaugmentation1 = this_augmentation.augmentation1
            subaug1arg1 = this_augmentation.aug1arg1
            subaug1arg2 = this_augmentation.aug1arg2
            subaug1arg3 = this_augmentation.aug1arg3
            subaug1arg4 = this_augmentation.aug1arg4
            subaug1arg5 = this_augmentation.aug1arg5
            subaugmentation2 = this_augmentation.augmentation2
            subaug2arg1 = this_augmentation.aug2arg1
            subaug2arg2 = this_augmentation.aug2arg2
            subaug2arg3 = this_augmentation.aug2arg3
            subaug2arg4 = this_augmentation.aug2arg4
            subaug2arg5 = this_augmentation.aug2arg5
            subpercentage_augment_used = this_augmentation.percentage_used

            if composition_method == 1:
                if score_this_focus != highest_score:
                    continue
                else:
                    noop = 1
                    #print(f"highest score has {augment} with {score_this_focus}")
            elif composition_method == 2 or composition_method == 3:
                subpercentage_augment_used = (score_this_focus/total_score)*100
                if subpercentage_augment_used < 0.1: continue #else, there would be the possibility to not return any data and break the script

            this_augmented_train_basis_data, this_train_target_augmented = do_the_augmentations_1(trainig_datapath, target_channels_string, subaugmentation1, subaug1arg1, subaug1arg2, subaug1arg3, subaug1arg4, subaug1arg5, augmented_train_basis_data, train_target_augmented, train_target_augmented,model)
            this_augmented_train_basis_data, this_train_target_augmented = do_the_augmentations_2(trainig_datapath, target_channels_string, subaugmentation2, subaug2arg1, subaug2arg2, subaug2arg3, subaug2arg4, subaug2arg5, augmented_train_basis_data, train_target_augmented,model)
            if augment_before_va == True:
                this_train_data_augmented = calculate_derivatives(this_augmented_train_basis_data)

            
            this_train_data_ML_vector_augmented = create_full_ml_vector_optimized(this_train_data_augmented)
            this_train_target_augmented = this_train_target_augmented[:,:-CUTOFF]

            #DOING AUGMENTATION, THAT HAPPENS AFTER CREATION OF THE VECTORS
            this_train_data_ML_vector_augmented, this_train_target_augmented = do_augmentations_after_vectorization(augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4,aug1arg5,trainig_datapaths, this_train_data_ML_vector_augmented, this_train_target_augmented, train_target)

            #SAMPLE VECTOR HERE
            this_train_data_ML_vector_augmented, this_train_target_augmented = sample_ML_arrays(this_train_data_ML_vector_augmented, this_train_target_augmented, subpercentage_augment_used)

            list_full_train_data.append(this_train_data_ML_vector_augmented)
            list_full_train_target.append(this_train_target_augmented)
            #if augment != "MSE_threshold":
            #    DEBUGaugmented_train_basis_data = np.vstack(list_full_train_data)
            #    DEBUGtrain_target_augmented = np.concatenate(list_full_train_target).reshape(1, -1)
            #    print(f"after augment {augment}: DEBUGaugmented_train_basis_data: {DEBUGaugmented_train_basis_data.shape} DEBUGtrain_target_augmented {DEBUGtrain_target_augmented.shape}")
        if total_score > 0:
            train_data_ML_vector_augmented = np.concatenate(list_full_train_data, axis=0)
            #print(f"merged lsit: augmented_train_basis_data {augmented_train_basis_data.shape}")
            train_target_augmented = np.concatenate(list_full_train_target,axis=1)#.reshape(1, -1)
        else:
            train_data_ML_vector_augmented = np.empty((0,train_data_ML_vector.shape[1]))
            train_target_augmented = np.empty((1,0))


        #print(f"All_augments_by_score returning augmented_train_basis_data {augmented_train_basis_data.shape} train_target_augmented {train_target_augmented.shape}")
        
    else:
        #print(f"\nelse\n\n")
        #print(f"do_get_train_data_and_targetvectors pre augment returning augmented_train_basis_data {augmented_train_basis_data.shape} train_target_augmented {train_target_augmented.shape}")
        augmented_train_basis_data, train_target_augmented = do_the_augmentations_1(trainig_datapath, target_channels_string, augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4,aug1arg5, augmented_train_basis_data, train_target_augmented, train_target_augmented,model)
        augmented_train_basis_data, train_target_augmented = do_the_augmentations_2(trainig_datapath, target_channels_string, augmentation2, aug2arg1, aug2arg2, aug2arg3, aug2arg4,aug2arg5, augmented_train_basis_data, train_target_augmented,model)
        #print(f"do_get_train_data_and_targetvectors psot augment returning augmented_train_basis_data {augmented_train_basis_data.shape} train_target_augmented {train_target_augmented.shape}")
        if augment_before_va == True:
            train_data = calculate_derivatives(train_basis_data)
            train_data_augmented = calculate_derivatives(augmented_train_basis_data)

        ##START TESTS
        if DOTESTFORMULA == 1:
            train_data = transform_array(train_data)
            train_data_augmented = transform_array(train_data_augmented)

        ##END TESTS

        #GETTING ML VECTORS FROM THE ORIGINAL DATA AS WELL AS THE AUGMENTED DATA
        train_data_ML_vector = create_full_ml_vector_optimized(train_data) #optimized means just, that i changed the function to make it faster. The result is the same
        train_data_ML_vector_augmented = create_full_ml_vector_optimized(train_data_augmented)
        train_target = train_target[:,:-CUTOFF] 
        train_target_augmented = train_target_augmented[:,:-CUTOFF]
        #DOING AUGMENTATION, THAT HAPPENS AFTER CREATION OF THE VECTORS
        train_data_ML_vector_augmented, train_target_augmented = do_augmentations_after_vectorization(augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4,aug1arg5,trainig_datapaths, train_data_ML_vector_augmented, train_target_augmented, train_target)
    #limiting original Data if percentage_used is not 100

    if percentage_original_used != 100:
        #print(f"pre sampling train_data_ML_vector {train_data_ML_vector.shape} train_target {train_target.shape}")
        train_data_ML_vector, train_target = sample_arrays(train_data_ML_vector, train_target, percentage_original_used)
        #print(f"after sampling train_data_ML_vector {train_data_ML_vector.shape} train_target {train_target.shape}")
    
    if percentage_augmented_used != 100:
        train_data_ML_vector_augmented, train_target_augmented = sample_arrays(train_data_ML_vector_augmented, train_target_augmented, percentage_augmented_used)
    #MERGING NORMAL AND AUGMENTED DATA
    #print(f"pre merging orig and generated data: train_data_ML_vector_augmented {train_data_ML_vector_augmented.shape} train_data_ML_vector {train_data_ML_vector.shape} train_target_augmented {train_target_augmented.shape} train_target {train_target.shape}")
    if augmentation1 != "None":
        train_data_ML_vector = np.concatenate((train_data_ML_vector_augmented, train_data_ML_vector), axis=0)
        train_target = np.concatenate((train_target_augmented, train_target), axis=1)
    if augmentation1 == "LimitTrainingdata":
        train_data_ML_vector = train_data_ML_vector_augmented
        train_target = train_target_augmented
    #print(f"post merging orig and generated data: train_data_ML_vector_augmented {train_data_ML_vector_augmented.shape} train_data_ML_vector {train_data_ML_vector.shape} train_target_augmented {train_target_augmented.shape} train_target {train_target.shape}")
    #ADDING DATA FROM THIS RUN TO THE FINAL VECTORS, THAT WILL BE FED INTO THE NN
    #if full_train_data_ML_vector is None:
    #    full_train_data_ML_vector = train_data_ML_vector
    #    full_train_target = train_target
    #else:
    #    full_train_data_ML_vector = np.vstack((full_train_data_ML_vector, train_data_ML_vector))
    #    full_train_target = np.array([np.concatenate((full_train_target[0] ,train_target[0]))])
    #print(f"returning {train_data_ML_vector.shape} and {train_target[0].shape}") #returning (42127, 40) and (42127,)
    return train_data_ML_vector, train_target[0]
    list_full_train_data_ML_vector.append(train_data_ML_vector)
    list_full_train_target.append(train_target[0])

def wrapper_function(i, trainig_datapaths, augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4, aug1arg5, augmentation2, aug2arg1, aug2arg2, aug2arg3, aug2arg4, aug2arg5, augment_before_va, target_channels_string,percentage_original_used,percentage_augment_used,model):
    return do_get_train_data_and_targetvectors(i, trainig_datapaths, augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4, aug1arg5, augmentation2, aug2arg1, aug2arg2, aug2arg3, aug2arg4, aug2arg5, augment_before_va, target_channels_string,percentage_original_used,percentage_augment_used,model)

def read_train_eval(data_params, aug_params, ML_params, augment_before_va, additional_descriptor = "", do_plot = False): 
    trainig_datapaths = data_params.trainig_datapaths
    validation_datapaths = data_params.validation_datapaths
    testing_datapaths = data_params.testing_datapaths
    percentage_original_used = data_params.percentage_used
    experiment_name = data_params.name
    #training_channels = data_params.training_channels
    #target_channels = data_params.target_channels
    #smoothing = data_params.smoothing
    #modulo_split = data_params.modulo_split
    augmentation1 = aug_params.augmentation1
    aug1arg1 = aug_params.aug1arg1
    aug1arg2 = aug_params.aug1arg2
    aug1arg3 = aug_params.aug1arg3
    aug1arg4 = aug_params.aug1arg4
    aug1arg5 = aug_params.aug1arg5
    augmentation2 = aug_params.augmentation2
    aug2arg1 = aug_params.aug2arg1
    aug2arg2 = aug_params.aug2arg2
    aug2arg3 = aug_params.aug2arg3
    aug2arg4 = aug_params.aug2arg4
    aug2arg5 = aug_params.aug2arg5
    percentage_augment_used = aug_params.percentage_used
    model = ML_params.model

    
    #GET DESCRIPTIVE STRING

    trainig_datapaths_string, validation_datapaths_string, training_channels_string, target_channels_string, descriptor_string_data, descriptor_string = generate_descriptor_strings(data_params, aug_params,ML_params,additional_descriptor,augment_before_va)
    

    list_full_train_data_ML_vector = []
    list_full_train_target = []
    results = Parallel(n_jobs=1)(delayed(wrapper_function)(i, trainig_datapaths, augmentation1, aug1arg1, aug1arg2, aug1arg3, aug1arg4, aug1arg5, augmentation2, aug2arg1, aug2arg2, aug2arg3, aug2arg4, aug2arg5, augment_before_va, target_channels_string,percentage_original_used,percentage_augment_used,model) for i in trainig_datapaths)
    for this_train_data_ML_vector, this_train_target in results:
        list_full_train_data_ML_vector.append(this_train_data_ML_vector)
        list_full_train_target.append(this_train_target)
                                                                     
    full_train_data_ML_vector = np.vstack(list_full_train_data_ML_vector)
    full_train_target = np.concatenate(list_full_train_target).reshape(1, -1)

    #GETTING VALIDATION DATA TO RIGHT FORMAT:
    fulltestdata = read_fulldata(validation_datapaths[0])
    fullvaldata = read_fulldata(testing_datapaths[0])
    test_basis_data = fulltestdata[:-4,:]
    val_basis_data = fullvaldata[:-4,:]
    test_basis_data = moving_average(test_basis_data, window_size=WINDOWSIZE) #eigentlich nicht notwendig
    val_basis_data = moving_average(val_basis_data, window_size=WINDOWSIZE) #eigentlich nicht notwendig
    test_target = get_target_current(target_channels_string, fulltestdata[-4:,:]).reshape((1, -1))
    val_target = get_target_current(target_channels_string, fullvaldata[-4:,:]).reshape((1, -1))
    test_target = test_target[:,:-CUTOFF] #shape need to be schanged
    val_target = val_target[:,:-CUTOFF] #shape need to be schanged
    test_target = np.abs(test_target)
    val_target = np.abs(val_target)
    test_data = calculate_derivatives(test_basis_data)
    val_data = calculate_derivatives(val_basis_data)

    if DOTESTFORMULA == 1:
        test_data = transform_array(test_data)
        val_data = transform_array(val_data)

    test_data_ML_vector = create_full_ml_vector_optimized(test_data)
    val_data_ML_vector = create_full_ml_vector_optimized(val_data)


    full_train_target = moving_average(full_train_target, window_size=WINDOWSIZE)
    test_target = moving_average(test_target, window_size=WINDOWSIZE)
    val_target = moving_average(val_target, window_size=WINDOWSIZE)
    
    #TRAIN THE ML MODEL
    if augmentation1 == "All_augments_by_score" and aug1arg5 == 1:
        save_traindata_path = CURRENT_DIRECTORY + rf"\Datensaetze\temp_for_augment\results\augmented_trainingdata_full_train_data_ML_vector.pkl"
        save_traintarget_path = CURRENT_DIRECTORY + rf"\Datensaetze\temp_for_augment\results\augmented_trainingdata_full_train_target.pkl"
        with open(save_traindata_path, 'wb') as f:
            pickle.dump(full_train_data_ML_vector, f)
        with open(save_traintarget_path, 'wb') as f:
            pickle.dump(full_train_target, f)
        #print(f"full_train_data_ML_vector {full_train_data_ML_vector.shape} full_train_target {full_train_target.shape}")
        predictions = test_target[0]
    else:
        model_string = str(model + "_"+target_channels_string[4:].capitalize()+"_"+experiment_name)
        predictions = do_the_training(model, ML_params, full_train_data_ML_vector,val_data_ML_vector, test_data_ML_vector, full_train_target,val_target,test_target,model_string)
    #return predictions
    
    #EVALUATE
    pred_parameters1 = predictions
    true_parameters1 = test_target[0]

    true_current_total = np.sum((true_parameters1)) # np.sum(np.absolute(true_parameters1))
    pred_current_total = np.sum((pred_parameters1)) # np.sum(np.absolute(pred_parameters1))#DSIABLED since deviation is always only abs
    
    ####cumulative_pred = np.cumsum(pred_parameters1)
    ####cumulative_true = np.cumsum(true_parameters1)
    ####fig, ax1 = plt.subplots()  # Initialize figure and primary axis
    ##### Plot original signals on primary y-axis
    ####ax1.plot(pred_parameters1, label='Predicted Parameters', color='blue')
    ####ax1.plot(true_parameters1, label='True Parameters', color='green')
    ####ax1.set_xlabel('Index')
    ####ax1.set_ylabel('Value', color='black')
    ####ax1.tick_params(axis='y', labelcolor='black')
    ##### Create a secondary y-axis for the cumulative sums
    ####ax2 = ax1.twinx()
    ####ax2.plot(cumulative_pred, label='Cumulative Predicted', linestyle='--', color='orange')
    ####ax2.plot(cumulative_true, label='Cumulative True', linestyle='--', color='red')
    ####ax2.set_ylabel('Cumulative Sum', color='black')
    ####ax2.tick_params(axis='y', labelcolor='black')
    ##### Combined legend for all lines
    ####lines, labels = ax1.get_legend_handles_labels()
    ####lines2, labels2 = ax2.get_legend_handles_labels()
    ####ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    ####plt.title('Parameter Predictions and Cumulative Sums')
    ####plt.show()

    #print(f"this_predictions {pred_parameters1.shape}")
    #print(f"this_true_values {true_parameters1.shape}")
    #print(f"finished DTW")

    sum_diff = -1#np.sum(np.abs(true_parameters1 - pred_parameters1))#DISABLED since never checked
    mae = mean_absolute_error(true_parameters1, pred_parameters1)
    mse = mean_squared_error(true_parameters1, pred_parameters1)
    rmse = -1#np.sqrt(mean_squared_error(true_parameters1, pred_parameters1))#DISABLED since never used
    deviation = round((1-(pred_current_total/true_current_total))*100,2)
    abs_deviation = -1#round((1-(abs_pred_current_total_1/abs_true_current_total_1))*100,2)#DSIABLED since deviation is always only abs

    #PLOT IT
    if do_plot == True:
        x_val = np.linspace(1,len(true_parameters1),len(true_parameters1))
        x_val = x_val/50
        plt.xlabel("Time in s")
        plt.ylabel("Current in A")
        
        # Plot the predicted values
        plt.title(descriptor_string)
        plt.scatter(x_val,pred_parameters1, s=1, label='Predicted')
        plt.scatter(x_val,true_parameters1, s=1, label='True')

        plt.legend()
        #print(descriptor_string)
        plt.show()
    return sum_diff, mae, mse, rmse, deviation, abs_deviation, descriptor_string, pred_parameters1.T, true_current_total, pred_current_total, true_parameters1

def run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations,augment_before_va, do_save_log, do_plot = False, plot_average_predictions = False): #TODO: Save data

    #THE FOLLOWING WAS ONLY FOR THE PRESENTATION FOR IRIS
    run_testfunctions_start = time.time()
    '''will run read_train_eval iteration times and return the average accuracy. Also saves the data'''
    this_sum_diffs = []
    this_MSEs = []
    this_deviations = []
    this_pred_current_totals = []
    this_true_current_totals = []
    descriptor_string = "descriptor_string"
    resultlist_integrated_deviations = []
    resultlist_MSEs = []
    resultlist_ABSULUTEABWEICHUNG = []
    resultlist_Q_meas = []
    resultlist_Q_est = []
    resultlist_avg_cv = []
    
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    deviation_log_path = DEVIATION_LOG_PATH#current_directory+"\\Auswertung\\deviation_log.csv"
    test_results_path = TEST_RESULTS_PATH #current_directory+"\\Auswertung\\test_results.csv"
    
    predictions = None
    actual_values = None
    #for i in range(iterations):
    #for i in tqdm(range(iterations), desc="Processing", ascii=True):
    desc_string = str(ML_params_test.model) + " "+str(data_params_test.name)+" "+str(data_params_test.target_channels[0])
    with tqdm(range(iterations), desc=desc_string, ascii=True) as pbar:
        for i in pbar:
            #readtraineval_start = time.time()
            this_sum_diff, this_mae, this_mse, this_rmse, this_deviation, this_abs_deviation, descriptor_string, this_predictions,true_current_total_1, pred_current_total_1, this_true_values = read_train_eval(data_params_test, aug_params_test, ML_params_test, augment_before_va,additional_descriptor,  do_plot)
            #this_predictions = this_predictions.T
            #print(f"in tqdm this_predictions {this_predictions.shape}") #NN (1, 24675) <- with that everything worx
            if predictions is None:
                predictions = this_predictions
                actual_values = this_true_values
            else:
                predictions = np.vstack((predictions,this_predictions))
                actual_values = np.vstack((actual_values,this_true_values))
            #print(f"this_predictions {this_predictions.shape}")
            #print(f"this_true_values {this_true_values.shape}")
            #print(f"Run iteration {i} of readtraineval in {round(time.time()-readtraineval_start,2)} seconds")
            this_sum_diffs.append(this_sum_diff)
            this_MSEs.append(this_mse)
            this_deviations.append(this_deviation)
            this_pred_current_totals.append(pred_current_total_1)
            this_true_current_totals.append(true_current_total_1)
            sum_this_pred_current_totals = sum(this_pred_current_totals)
            sum_this_true_current_totals = sum(this_true_current_totals)
            this_integral_deviataon = abs(round(((sum_this_pred_current_totals-sum_this_true_current_totals)/sum_this_true_current_totals)*100,4)) #abs(1-(sum_this_true_current_totals/sum_this_pred_current_totals))*100
            this_cv = (np.std(np.array(this_pred_current_totals))/np.mean(np.array(this_pred_current_totals)))*100

            #FOR RESULTS
            resultlist_integrated_deviations.append(this_deviation)
            resultlist_MSEs.append(this_mse)
            resultlist_ABSULUTEABWEICHUNG.append(-1) #Für jeden punkt abweichung in % berechnen, dann durchschnitt bilden
            resultlist_Q_meas.append(true_current_total_1)
            resultlist_Q_est.append(pred_current_total_1)
            resultlist_avg_cv.append(this_cv)

            avg_mse = round(sum(resultlist_MSEs)/len(resultlist_MSEs),4)
            avg_dev = abs(round(sum(resultlist_integrated_deviations)/len(resultlist_MSEs),2))

            now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            row_for_csv = [str(now),str(this_sum_diff),str(this_mae),str(this_mse),str(this_cv),str(this_deviation),str(this_abs_deviation),str(true_current_total_1),str(pred_current_total_1), str(descriptor_string)]
            if do_save_log == True:
                with open(deviation_log_path, 'a', newline='') as csvfile:
                    # creating a csv writer object
                    csvwriter = csv.writer(csvfile, delimiter=',')
                    csvwriter.writerow(row_for_csv)
            pbar.set_postfix({"avg_integral_dev ": f" {round(this_integral_deviataon,4)} % cv: {round(this_cv,2)} % avg_MSE: {avg_mse} avg_DEV: {avg_dev}"}, refresh=True)
            if i > 1 and this_cv < 0.1: break #breaks if CV is small enough after 3 iterations 
            if i > 3 and this_cv < 10: break #breaks if CV is small enough after 5 iterations 

    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    nr_of_values = len(resultlist_MSEs)
    result_time_per_it = round((time.time()-run_testfunctions_start)/nr_of_values,1)
    result_MSE = round(sum(resultlist_MSEs)/nr_of_values,6)
    std_MSEs = round(np.std(resultlist_MSEs),6)
    result_ABSULUTEABWEICHUNG = round(sum(resultlist_ABSULUTEABWEICHUNG)/nr_of_values,4) 
    result_Q_meas = round(sum(resultlist_Q_meas)/nr_of_values,1)
    result_Q_est = round(sum(resultlist_Q_est)/nr_of_values,1)
    result_deviation_percentage = abs(round(((result_Q_est-result_Q_meas)/result_Q_meas)*100,4))
    std_deviation_percentage = abs(round((((result_Q_est+np.std(resultlist_Q_est)-result_Q_meas)/result_Q_meas)*100)-result_deviation_percentage,4))

    result_CV = round(sum(resultlist_avg_cv)/nr_of_values,4)
    row_for_results = [str(now),str(nr_of_values), str(result_time_per_it), str(result_deviation_percentage), str(std_deviation_percentage), str(result_MSE), str(std_MSEs), str(result_Q_meas),str(result_Q_est),str(result_CV), str(descriptor_string)]
    row_for_results = [f"{x:.6f}" if isinstance(x, float) else x for x in row_for_results]
    if do_save_log == True:
        with open(test_results_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(row_for_results)

    if plot_average_predictions == True:
        #Thsi will also generate MSE_values
        mse_folder = r"C:\Users\mauma\Google Drive\Projekte\Master\Masterarbeit\Software\Datensaetze\MSE_values"
        print(f"predictions {predictions.shape}")

        plot_multiple_predictions(predictions)

        ####print(f"actual_values {actual_values.shape}")
        mean_predictions = predictions.mean(axis = 0)
        mean_actual_values = actual_values.mean(axis = 0)
        ####print(f"plotmean data_params_test.target_channels {data_params_test.target_channels[0]}")
        ####print(f"plotmean data_params_test.trainig_datapaths {data_params_test.trainig_datapaths[0]}")
        ####print(f"mean_predictions {mean_predictions.shape}")
        ####print(f"mean_actual_values {mean_actual_values.shape}")
        this_MSEs = calculate_moving_mse(mean_predictions, mean_actual_values)
        #print(f"this_MSEs {this_MSEs.shape}")
        this_MSEs = this_MSEs*40
        mse_savefile = mse_folder+"\\MSEs_"+data_params_test.validation_datapaths[0]+"_"+data_params_test.target_channels[0]+".pkl"
        print(mse_savefile)

        with open(mse_savefile, 'wb') as f:
            pickle.dump(this_MSEs, f)

        #'''
        mean_predictions = predictions.mean(axis = 0)
        mean_actual_values = actual_values.mean(axis = 0)
        x_val = np.linspace(1,len(mean_predictions),len(mean_predictions))
        x_val = x_val/50
        #plt.xlabel("Time in s")
        #plt.ylabel("Current in A")
        ## Plot the predicted values
        #plt.title(descriptor_string+"\n Mean over "+str(iterations)+" iterations ")
        #plt.scatter(x_val,mean_predictions, s=1, label='Mean Predicted')
        #plt.scatter(x_val,mean_actual_values, s=1, label='True')
        #plt.scatter(x_val,this_MSEs, s=1, label='MSE')
        #plt.legend()
        #plt.show()
        #'''
        #plot
    #print(f"finished run_testfunction. this MSEs: {this_MSEs} with average {np.average(np.array(this_MSEs))}"
        
    if predictions.ndim == 1:
        mean_predictions = predictions
    else:
        mean_predictions = predictions.mean(axis = 0)
    
    if actual_values.ndim == 1:
        mean_actual_values = actual_values
    else:
        mean_actual_values = actual_values.mean(axis = 0)

    #mean_actual_values = actual_values.mean(axis = 0)
    result_stddev = ((result_CV*0.01*result_Q_est)/result_Q_meas)*100
    print(f"Dev % {result_deviation_percentage}, MSE {result_MSE}")
    return [result_deviation_percentage,std_deviation_percentage], [result_MSE, std_MSEs], mean_predictions[:-10], mean_actual_values[:-10]

def run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = True, do_save_log=False, do_plot = False, plot_average_predictions = False, argfinder_mode =False):
    '''will run the test for all 4 models and all 4 currents'''
    target_currents = [['cur_x'] ,['cur_y'],['cur_z'],['cur_sp']]
    test_results_path = TEST_RESULTS_PATH 
    row_for_results = [f"{data_params_test.name} with Augmentation1: {aug_params_test.augmentation1} aug1arg1: {aug_params_test.aug1arg1} aug1arg2: {aug_params_test.aug1arg2} aug1arg3: {aug_params_test.aug1arg3} aug1arg4: {aug_params_test.aug1arg4} aug1arg5: {aug_params_test.aug1arg5}, , , , , , , , "]
    if do_save_log == True:
        with open(test_results_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(row_for_results)
            #csvwriter.writerow(["NN_Cheap, , , , , , , , "])# csvwriter.writerow(["RF_Normal, , , , , , , , "]) #csvwriter.writerow(["NN_Cheap, , , , , , , , "])
            csvwriter.writerow(["RF_Normal, , , , , , , , "]) #csvwriter.writerow(["NN_Cheap, , , , , , , , "])
            #csvwriter.writerow(["NN_Cheap, , , , , , , , "])
    
    TEST_ITERATIONS = 10 #
    ML_params_test = RF_Normal()
    for this_current in target_currents:
        data_params_test.target_channels = this_current#['cur_x'] 
        #print(f"Training {ML_params_test.model} on {data_params_test.target_channels[0]} Augmentation: {aug_params_test.augmentation1}  with aug1arg1: {aug_params_test.aug1arg1} and aug1arg2: {aug_params_test.aug1arg2} and aug1arg3: {aug_params_test.aug1arg3}")
        run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = TEST_ITERATIONS, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions)
    if argfinder_mode == True: return
    #print(f"\n\nALARM, ALARM, you forgot to change it back to start with RF_Normal instead of NN_Cheap")
    
    ML_params_test = RF_Cheap()
    if do_save_log == True:
        with open(test_results_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(["RF_Cheap, , , , , , , , "])
    for this_current in target_currents:
        data_params_test.target_channels = this_current#['cur_x'] 
        #print(f"Training {ML_params_test.model} on {data_params_test.target_channels[0]} Augmentation: {aug_params_test.augmentation1}  with aug1arg1: {aug_params_test.aug1arg1} and aug1arg2: {aug_params_test.aug1arg2} and aug1arg3: {aug_params_test.aug1arg3}")
        run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = TEST_ITERATIONS, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions)
    
    TEST_ITERATIONS = 10 #for the NN, since they have quite some variance
    ML_params_test = NN_Normal()
    if do_save_log == True:
        with open(test_results_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(["NN_Normal, , , , , , , , "])
    for this_current in target_currents:
        data_params_test.target_channels = this_current#['cur_x'] 
        #print(f"Training {ML_params_test.model} on {data_params_test.target_channels[0]} Augmentation: {aug_params_test.augmentation1}  with aug1arg1: {aug_params_test.aug1arg1} and aug1arg2: {aug_params_test.aug1arg2} and aug1arg3: {aug_params_test.aug1arg3}")
        run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = TEST_ITERATIONS, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions)
    ML_params_test = NN_Cheap()
    if do_save_log == True:
        with open(test_results_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(["NN_Cheap, , , , , , , , "])
    for this_current in target_currents:
        data_params_test.target_channels = this_current# 
        #print(f"Training {ML_params_test.model} on {data_params_test.target_channels[0]} Augmentation: {aug_params_test.augmentation1}  with aug1arg1: {aug_params_test.aug1arg1} and aug1arg2: {aug_params_test.aug1arg2} and aug1arg3: {aug_params_test.aug1arg3}")
        run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = TEST_ITERATIONS, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions)

def run_full_tests_aircut_CMX(aug_params_test, additional_descriptor, augment_before_va = True, do_save_log=False, do_plot = False, plot_average_predictions = False, argfinder_mode = False):
    '''runns the 16 different tests on Versuch 1a, 1b, 2a, 3a'''
    data_params_test = data_Versuch_1_CMX_aircut()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_2_CMX_aircut()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_3_CMX_aircut()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_4_CMX_aircut()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)

def run_full_tests_aircut_I40(aug_params_test, additional_descriptor, augment_before_va = True, do_save_log=False, do_plot = False, plot_average_predictions = False, argfinder_mode = False):
    '''runns the 16 different tests on Versuch 1a, 1b, 2a, 3a'''
    data_params_test = data_Versuch_1_I40_aircut()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_2_I40_aircut()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_3_I40_aircut()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_4_I40_aircut()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)

def run_full_tests_process_I40(aug_params_test, additional_descriptor, augment_before_va = True, do_save_log=False, do_plot = False, plot_average_predictions = False, argfinder_mode = False):
    '''runns the 16 different tests on Versuch 1a, 1b, 2a, 3a'''
    #data_params_test = data_Versuch_1_I40_prozess()
    #run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_2_I40_prozess()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    #data_params_test = data_Versuch_3_I40_prozess()
    #run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_4_I40_prozess()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)


def run_full_tests_process_CMX(aug_params_test, additional_descriptor, augment_before_va = True, do_save_log=False, do_plot = False, plot_average_predictions = False, argfinder_mode = False):
    '''runns the 16 different tests on Versuch 1a, 1b, 2a, 3a'''
    #data_params_test = data_Versuch_1_prozess()
    #run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_2_CMX_prozess()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    #data_params_test = data_Versuch_3_prozess()
    #run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)
    data_params_test = data_Versuch_4_CMX_prozess()
    run_full_test(data_params_test, aug_params_test, additional_descriptor, augment_before_va = augment_before_va, do_save_log=do_save_log, do_plot = do_plot, plot_average_predictions = plot_average_predictions, argfinder_mode = argfinder_mode)

#ML_params_test = RF_Normal() 
additional_descriptor = ""#"optimized_percentages" #see line 1384

#aug_params_test = NoAugment()# Overtrain_bad_MSEs()#smogn_augmentation()#TimeWarp() #smogn_augmentation()#Noise()#TimeVAEgenerated
def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


DO_SAVE_LOG = False
ITERATIONS = 10
DOTESTFORMULA = 0

aug_params_test = NoAugment() 
data_params_test = data_Versuch_3_CMX_prozess() 
ML_params_test = NN_Cheap() 
data_params_test.target_channels = ['cur_x']
run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = ITERATIONS,augment_before_va = True, do_save_log = False, do_plot = False, plot_average_predictions = False)
#data_params_test.target_channels = ['cur_y']
#run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = ITERATIONS,augment_before_va = True, do_save_log = False, do_plot = False, plot_average_predictions = False)
#data_params_test.target_channels = ['cur_z']
#run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = ITERATIONS,augment_before_va = True, do_save_log = False, do_plot = False, plot_average_predictions = False)
#data_params_test.target_channels = ['cur_sp']
#run_testfunctions(data_params_test, aug_params_test, ML_params_test, additional_descriptor, iterations = ITERATIONS,augment_before_va = True, do_save_log = False, do_plot = False, plot_average_predictions = False)

