
#tf.compat.v1.enable_v2_behavior()
import tensorflow as tf
#tf.compat.v1.enable_v2_behavior()
from baca import *
from deeplearning_separation import *

import numpy as np

os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 12})
path='../data/mix'
sr,mix, length=bacasignal(path)

#target saron
path='../data/saron'
sr,saron, p=bacasignal(path)
#insert 0 in saron if, there is no saron playing in mix
zero=np.zeros(length[1])
saron.insert(1,zero)

#target demung
path='../data/demung'
sr, demung, p=bacasignal(path)
zero=np.zeros(length[2])
demung.insert(2,zero)

#target peking
path='../data/peking'
sr, peking, p=bacasignal(path)
zero=np.zeros(length[0])
peking.insert(0,zero)

#target slenthem
path='../data/slenthem'
sr,slenthem, p=bacasignal(path)
zero0=np.zeros(length[0])
slenthem.insert(0,zero0)
zero1=np.zeros(length[1])
slenthem.insert(1,zero1)
zero2=np.zeros(length[2])
slenthem.insert(2,zero2)
zero3=np.zeros(length[3])
slenthem.insert(3,zero3)

zero5=np.zeros(length[5])
slenthem.insert(5,zero5)
zero7=np.zeros(length[7])
slenthem.insert(7,zero7)
#target boanang barung
path='../data/bonangbarung'
sr,bonangbarung, p=bacasignal(path)
zero4=np.zeros(length[4])

bonangbarung.insert(0,zero0)
bonangbarung.insert(1,zero1)
bonangbarung.insert(2,zero2)
bonangbarung.insert(3,zero3)
bonangbarung.insert(4,zero4)



#target bonang penerus
path='../data/bonangpenerus'
sr,bonangpenerus, p=bacasignal(path)
bonangpenerus.insert(0,zero0)
bonangpenerus.insert(1,zero1)
bonangpenerus.insert(2,zero2)
bonangpenerus.insert(3,zero3)
bonangpenerus.insert(4,zero4)


#target gong
path='../data/gong'
sr,gong, p=bacasignal(path)
zero5=np.zeros(length[5])
zero6=np.zeros(length[6])

gong.insert(0,zero0)
gong.insert(1,zero1)
gong.insert(2,zero2)
gong.insert(3,zero3)
gong.insert(4,zero4)
gong.insert(5,zero5)
gong.insert(6,zero6)

#target kendhang
path='../data/kendhang'
sr,kendhang, p=bacasignal(path)

zero7=np.zeros(length[7])
zero8=np.zeros(length[8])

kendhang.insert(0,zero0)
kendhang.insert(1,zero1)
kendhang.insert(2,zero2)
kendhang.insert(3,zero3)
kendhang.insert(4,zero4)
kendhang.insert(5,zero5)
kendhang.insert(6, zero6)
kendhang.insert(7, zero7)
kendhang.insert(8,zero8)

#target class
kelas=np.array([[1,1,0,0,0,0,0,0],[0,1,1,0,0,0,0,0],[1,0,1,0,0,0,0,0],
                [1,1,1,0,0,0,0,0],[1,1,1,0,0,1,0,0],[1,1,1,1,1,0,0,0],
                [1,1,1,1,1,1,0,0],[1,1,1,1,1,0,1,0],[1,1,1,1,1,1,1,0],
                [1,1,1,1,1,1,1,1]])
kelas_saron=np.array([[1],[0],[1],[1],[1],[1],[1],[1],[1],[1]])
kelas_demung=np.array([[1],[1],[0],[1],[1],[1],[1],[1],[1],[1]])
kelas_peking=np.array([[0],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
kelas_bonangbarung=np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])
kelas_bonangpenerus=np.array([[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]])
kelas_slenthem=np.array([[0],[0],[0],[0],[1],[0],[1],[0],[1],[1]])
kelas_gong=np.array([[0],[0],[0],[0],[0],[0],[0],[1],[1],[1]])
kelas_kendhang=np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]])

#ambil data training lagu ke 1, 5, 6, 9, 10 atau index ke 0,4,5,8,9
#semua data train
# mix_train=[mix[0],mix[4],mix[5],mix[8],mix[9]]
# saron_train=[saron[0],saron[4],saron[5],saron[8],saron[9]]
# demung_train=[demung[0],demung[4],demung[5],demung[8],demung[9]]
# peking_train=[peking[0],peking[4],peking[5],peking[8],peking[9]]
# slenthem_train=[slenthem[0],slenthem[4],slenthem[5],slenthem[8],slenthem[9]]
# bonangbarung_train=[bonangbarung[0],bonangbarung[4],bonangbarung[5], bonangbarung[8],bonangbarung[9]]
# bonangpenerus_train=[bonangpenerus[0], bonangpenerus[4], bonangpenerus[5], bonangpenerus[8], bonangpenerus[9]]
# gong_train=[gong[0],gong[4],gong[5], gong[8], gong[9]]
# kendhang_train=[kendhang[0], kendhang[4], kendhang[5],kendhang[8], kendhang[9]]

# kelas_train=[kelas[0], kelas[4], kelas[5], kelas[8], kelas[9]]


#pakai data ke-2
mix_train=[mix[1]]
saron_train=[saron[1]]
demung_train=[demung[1]]
peking_train=[peking[1]]
slenthem_train=[slenthem[1]]
bonangbarung_train=[bonangbarung[1]]
bonangpenerus_train=[bonangpenerus[1]]
gong_train=[gong[1]]
kendhang_train=[kendhang[1]]
#kelas_train=[kelas[1]]
p_saron=[kelas_saron[1]]
p_demung=[kelas_demung[1]]
p_peking=[kelas_peking[1]]
p_bonangbarung=[kelas_bonangbarung[1]]
p_bonangpenerus=[kelas_bonangpenerus[1]]
p_slenthem=[kelas_slenthem[1]]
p_gong=[kelas_slenthem[1]]
p_kendhang=[kelas_kendhang[1]]
#windowing

win=1026 #93ms
hop=512
numpad=5 #number of padding
# win=44 #1ms
# hop=21
xtrain,ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang, s=windowing(mix_train,saron_train,demung_train, peking_train, bonangbarung_train, bonangpenerus_train, slenthem_train, gong_train, kendhang_train,win,hop)


#memastikan KELAS PADA WINDOWING
kls_saron,kls_demung,kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang=matchkelas(ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang,s, p_saron,p_demung,p_peking,p_bonangbarung,p_bonangpenerus,p_slenthem,p_gong,p_kendhang)

#fourier

fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang=fourier(xtrain,ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang)
#softmask by saya
ftrain,csaron,cdemung,cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang,timestamp,feature=binarymask(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang,numpad)

#hasil save to h5
savefile(ftrain,csaron,cdemung,cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang,timestamp,feature, kls_saron,kls_demung, kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang)

#load file
# path='../matrix/1ms/mattrain.h5'
# ftrain, csaron, cdemung, cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang=loadh5file(path)
#original binary mask
#ftrain,csaron,cdemung,cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang,timestamp,feature=aslibinarymask(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang)

#realsoftmask
#ftrain_s,csaron_s,cdemung_s,cpeking_s, cbonangbarung_s, cbonangpenerus_s, cslenthem_s, cgong_s, ckendhang_s,timestamp_s,numfeature_s=softmask(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang)

#binary mask da`ri paper Olga Slizovskaia, IEEE ACM 2021
#ftrain,csaron,cdemung,cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang,timestamp,feature=binarymask1(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang)


#model_softmask,history_softmask,pred_softmask=Unet(ftrain_s,csaron_s,cdemung_s,cpeking_s, cbonangbarung_s, cbonangpenerus_s, cslenthem_s, cgong_s, ckendhang_s,timestamp_s,numfeature_s, kls_saron,kls_demung, kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang)

model_93ms,history_93ms, pred_93ms=Unet(ftrain,csaron,cdemung,cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang,timestamp,feature, kls_saron,kls_demung, kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang)

#laod model
# from tensorflow.keras.models import load_model
# pred_01s= load_model('../model_save/unet.h5')
# pred=proposed.predict(ftrain)
#pred_01s=model_01s.predict(ftrain)

pred_saron=pred_93ms[0] #saron
pred_demung=pred_93ms[1] #demung
pred_peking=pred_93ms[2] #peking
pred_bonangbarung=pred_93ms[3]#bonangbarung

msaron,mdemung, mpeking,mbonangbarung=invers_mask(pred_saron,pred_demung, pred_peking,pred_bonangbarung,ftrain)

rs,rd,rp,rbb=invers_fourier(msaron,mdemung, mpeking,mbonangbarung,win,hop,i,numpad)

write_song(rs,rd,rp,rbb)
































