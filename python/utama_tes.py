#campuran demung dan peking

from baca import *
import tensorflow as tf
import numpy as np

os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
path='../data/dummy/mix'
sr,mix, length=bacasignal(path)
# mix=np.array(mix)
p=length[0]
# mix=np.resize(mix,[p,1])
#target saron

saron=np.zeros([p,1])
saron=saron.tolist()


#target demung
path='../data/dummy/demung'
sr, demung, p=bacasignal(path)
# tdemung=np.array(demung)
# p=np.size(demung,1)
# demung=np.resize(tdemung,[p,1])
#target peking
path='../data/dummy/peking'
sr, peking, p=bacasignal(path)
# peking=np.array(peking)
# p=np.size(peking,1)
# peking=np.resize(peking,[p,1])

#target slenthem
slenthem=np.zeros([p,1])
slenthem=slenthem.tolist()

#bonangbarung
bonangbarung=np.zeros([p,1])
bonangbarung=bonangbarung.tolist()

#target bonang penerus
bonangpenerus=np.zeros([p,1])
bonangpenerus=bonangpenerus.tolist()
#target gong
gong=np.zeros([p,1])
gong=gong.tolist()

#target kendhang
kendhang=np.zeros([p,1])
kendhang=kendhang.tolist()



#target class
kelas=np.array([1,1,0,0,0,0,0,0])
#kelas=np.resize(kelas,[1,8])

#ambil data training lagu ke 1, 5, 6, 9, 10 atau index ke 0,4,5,8,9



#windowing

xtrain,ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang, s=windowing(mix,saron,demung, peking, bonangbarung, bonangpenerus, slenthem, gong, kendhang)


#memastikan KELAS PADA WINDOWING
kelas_baru=matchkelas(ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang,kelas_train,s)

#fourier

fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang=fourier(xtrain,ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang)

#model,history,hsaron,hdemung,hpeking,hbonangbarung,hbonangpenerus,hslenthem,hgong,hkendhang=Unet(xtrain,ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang,kelas,sr)





































