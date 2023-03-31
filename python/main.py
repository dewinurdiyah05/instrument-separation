# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:44:47 2023

@author: user
"""

import tensorflow as tf
#tf.compat.v1.enable_v2_behavior()
from baca import *
from deeplearning_separation import *

import numpy as np

#read h5 file
#path file 1 ms windowing
path='../matrix/1ms/mat1ms.h5'
ftrain, csaron, cdemung, cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang, timestamp,feature, kls_saron, kls_demung, kls_peking, kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang=loadh5file(path)
model_1ms,history_1ms, pred_1ms=Unet(ftrain,csaron,cdemung,cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang,timestamp,feature, kls_saron,kls_demung, kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang)

#laod model
from tensorflow.keras.models import load_model
proposed= load_model('../save_model/1ms_normal.h5')
pred_1ms=proposed.predict(ftrain)

pred_saron=pred_1ms[0] #saron
pred_demung=pred_1ms[1] #demung
pred_peking=pred_1ms[2] #peking
pred_bonangbarung=pred_1ms[3]#bonangbarung

msaron,mdemung, mpeking,mbonangbarung=invers_mask(pred_saron,pred_demung, pred_peking,pred_bonangbarung,ftrain)
win=44
hop=21
numpad=5

i_demung,transdemung,i_demung,dm,hdemung,rawdemung,num_row,x_demung=invers_fourier(msaron,mdemung, mpeking,mbonangbarung,win,hop,numpad)
