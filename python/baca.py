# -*- coding: utf-8 -*-
"""
Created on Sat May 28 13:27:06 2022

@author: user
"""
#from skmultilearn.model_selection import iterative_train_test_split
import tensorflow as tf
from scipy.io.wavfile import write
import pandas as pd
import numpy as np
import librosa as lb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import glob
import os
#import torch
#from torch_audiomentations import Compose, Gain, PolarityInversion
#import soundfile as sf
import matplotlib.pyplot as plt
import wavio
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import h5py

def bacasignal(path):
     
    total=os.listdir(path)
 
    chanel1=[]
    chanel2=[]
    length=[]
 
    for filename in sorted(glob.glob(os.path.join(path, '*.wav'))):
        
        print(filename)
        
        data,sr = lb.load(filename,mono=False, sr=44100)
        
        
        #rescaling range 1 dan 0
        maks=np.max(data[0][0:])
        minim=np.min(data[0][0:])
        
        normal=np.nan_to_num((data[0][0:0]-minim)/(maks-minim))
       
        data1=data[0][0:]
        panjang=len(data1)
        # temp=np.zeros((panjang))
        
        # for i in range(panjang-1):
        #     if (data1[i:i+1]<0.009 and data1[i:i+1]>-0.009).any():
        #         temp[i:i+1]=0
        #     else :
        #         temp[i:i+1]=data1[i:i+1]
        
       
        chanel1.append(data1)
        #chanel2.append(data2)
        length.append(panjang)
        #waktu.append(t)
        # g.append(grup)
        # #aug.append(hsl)
        # k=k+1

    return  sr, chanel1,  length




def windowing(mix,saron,demung, peking, bonangbarung, bonangpenerus, slenthem, gong, kendhang,bagi,hop):
    p=len(mix)
    print(p)
    
    j=0
    Xmix=[]
    tsaron=[]
    tdemung=[]
    tpeking=[]
    tbonangbarung=[]
    tbonangpenerus=[]
    tslenthem=[]
    tgong=[]
    tkendhang=[]
    s=[]

    for k in range(p):
        total=np.floor(len(mix[k])/(bagi-hop))-1
        total=total.astype(int)
        j=j+total
        print('----------------',total)
        i=0
        s.append(total-1)
        for i in range(total):
            awal=hop*i
            akhir=hop*(i+1)+hop
            print(awal,akhir)
            Xmix.append(mix[k][awal:akhir])
            tsaron.append(saron[k][awal:akhir])
            tdemung.append(demung[k][awal:akhir])
            tpeking.append(peking[k][awal:akhir])
            tbonangbarung.append(bonangbarung[k][awal:akhir])
            tbonangpenerus.append(bonangpenerus[k][awal:akhir])
            tslenthem.append(slenthem[k][awal:akhir])
            tgong.append(gong[k][awal:akhir])
            tkendhang.append(kendhang[k][awal:akhir])
          
        
            
    xtrain=np.array(Xmix)
    ysaron=np.array(tsaron)
    ydemung=np.array(tdemung)
    ypeking=np.array(tpeking)
    ybonangbarung=np.array(tbonangbarung)
    ybonangpenerus=np.array(tbonangpenerus)
    yslenthem=np.array(tslenthem)
    ygong=np.array(tgong)
    ykendhang=np.array(tkendhang)

   
    
    # #coba invers sama gk?
    #yinv=inv.inverse_transform(yhot)
    
    return xtrain,ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang, s



def matchkelas(ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang,s, p_saron,p_demung,p_peking,p_bonangbarung,p_bonangpenerus,p_slenthem,p_gong,p_kendhang):    
    k_saron=np.array(p_saron)
    k_demung=np.array(p_demung)
    k_peking=np.array(p_peking)
    k_bonangbarung=np.array(p_bonangbarung)
    k_bonangpenerus=np.array(p_bonangpenerus)
    k_slenthem=np.array(p_slenthem)
    k_gong=np.array(p_gong)
    k_kendhang=np.array(p_kendhang)
    
    #numdata=np.size(ysaron,0)
    numsample=np.size(ysaron,1)
    kelas_saron=np.zeros([np.size(ysaron,0),1]) #1 adalah jumlah digit, 1 instrument
    kelas_demung=np.zeros([np.size(ysaron,0),1])
    kelas_peking=np.zeros([np.size(ysaron,0),1])
    kelas_bonangbarung=np.zeros([np.size(ysaron,0),1])
    kelas_bonangpenerus=np.zeros([np.size(ysaron,0),1])
    kelas_slenthem=np.zeros([np.size(ysaron,0),1])
    kelas_gong=np.zeros([np.size(ysaron,0),1])
    kelas_kendhang=np.zeros([np.size(ysaron,0),1])
    #kelas untuk saron
    num_kls=np.size(s,0)
    awal=0
    for k in range(num_kls):
        print("awal=",awal)
        akhir=awal+s[k]
        
        for awal in range(akhir):
            tsaron=0
            tdemung=0
            tpeking=0
            tbonangbarung=0
            tbonangpenerus=0
            tslenthem=0
            tgong=0
            tkendhang=0
            for j in range(numsample):
                tsaron=tsaron+(k_saron[k:k+1,0:1]*ysaron[awal][j])
                tdemung=tdemung+(k_demung[k:k+1,0:1]*ydemung[awal][j])
                tpeking=tpeking+(k_peking[k:k+1,0:1]*ypeking[awal][j])
                tbonangbarung=tbonangbarung+(k_bonangbarung[k:k+1,0:1]*ybonangbarung[awal][j])
                tbonangpenerus=tbonangpenerus+(k_bonangpenerus[k:k+1,0:1]*ybonangpenerus[awal][j])
                tslenthem=tslenthem+(k_slenthem[k:k+1,0:1]*yslenthem[awal][j])
                tgong=tgong+tslenthem+(k_gong[k:k+1,5:6]*ygong[awal][j])
                tkendhang=tkendhang+(k_kendhang[k:k+1,5:6]*ykendhang[awal][j])
                
            if (tsaron!=0).any():
                kelas_saron[awal:awal+1:,0:1]=1
            if (tdemung!=0).any():
                kelas_demung[awal:awal+1:,0:1]=1
            if (tpeking!=0).any():
                kelas_peking[awal:awal+1:,0:1]=1
            if (tbonangbarung!=0).any():
                kelas_bonangbarung[awal:awal+1:,0:1]=1
            if (tbonangpenerus!=0).any():
                kelas_bonangpenerus[awal:awal+1:,0:1]=1
            if (tslenthem!=0).any():
                kelas_slenthem[awal:awal+1:,0:1]=1
            if (tgong!=0).any():
                kelas_gong[awal:awal+1:,0:1]=1
            if (tkendhang!=0).any():
                kelas_gong[awal:awal+1:,7:8]=1
        
        print("akhir=",akhir)        
        awal=akhir       
   
    return kelas_saron,kelas_demung,kelas_peking,kelas_bonangbarung, kelas_bonangpenerus, kelas_slenthem, kelas_gong, kelas_kendhang

def fourier(xtrain,ysaron,ydemung,ypeking, ybonangbarung, ybonangpenerus, yslenthem, ygong, ykendhang):
    nsample=np.size(xtrain,0)
    numaudio=1
    numtimestep=np.size(xtrain,1)
    datamix=[]
    datasaron=[]
    datademung=[]
    datapeking=[]
    databonangbarung=[]
    databonangpenerus=[]
    dataslenthem=[]
    datagong=[]
    datakendhang=[]
    imaginer=[]
    
    for i in range(nsample):
        mix=np.resize(xtrain[i:i+1,:],[numaudio,numtimestep])
        fmix=tf.signal.rfft(mix)
        fomix=fmix.numpy()
        img=tf.math.imag(fmix)
        img_mix=img.numpy()
        imaginer.append(img_mix)
        x = tf.keras.backend.abs(fmix)
        train=x.numpy()
        datamix.append(train)
        
        saron=np.resize(ysaron[i:i+1,:],[numaudio,numtimestep])
        fsaron=tf.signal.rfft(saron)
        y1=tf.keras.backend.abs(fsaron)
        tsaron=y1.numpy()
        datasaron.append(tsaron)
        
        demung=np.resize(ydemung[i:i+1,:],[numaudio,numtimestep])
        fdemung=tf.signal.rfft(demung)
        y2=tf.keras.backend.abs(fdemung)
        tdemung=y2.numpy()
        datademung.append(tdemung)
        
        peking=np.resize(ypeking[i:i+1,:],[numaudio,numtimestep])
        fpeking=tf.signal.rfft(peking)
        y3=tf.keras.backend.abs(fpeking)
        tpeking=y3.numpy()
        datapeking.append(tpeking)
        
        bonangbarung=np.resize(ybonangbarung[i:i+1,:],[numaudio,numtimestep])
        fbonangbarung=tf.signal.rfft(bonangbarung)
        y4=tf.keras.backend.abs(fbonangbarung)
        tbonangbarung=y4.numpy()
        databonangbarung.append(tbonangbarung)
        
        
        bonangpenerus=np.resize(ybonangpenerus[i:i+1,:],[numaudio,numtimestep])
        fbonangpenerus=tf.signal.rfft(bonangpenerus)
        y5=tf.keras.backend.abs(fbonangpenerus)
        tbonangpenerus=y5.numpy()
        databonangpenerus.append(tbonangpenerus)
        
        slenthem=np.resize(yslenthem[i:i+1,:],[numaudio,numtimestep])
        fslenthem=tf.signal.rfft(slenthem)
        y6=tf.keras.backend.abs(fslenthem)
        tslenthem=y6.numpy()
        dataslenthem.append(tslenthem)
        
        gong=np.resize(ygong[i:i+1,:],[numaudio,numtimestep])
        fgong=tf.signal.rfft(gong)
        y7=tf.keras.backend.abs(fgong)
        tgong=y7.numpy()
        datagong.append(tgong)
        
        kendhang=np.resize(ykendhang[i:i+1,:],[numaudio,numtimestep])
        fkendhang=tf.signal.rfft(kendhang)
        y8=tf.keras.backend.abs(fkendhang)
        tkendhang=y8.numpy()
        datakendhang.append(tkendhang)
        
        
    # #convert to array
    ft=np.array(datamix)
    fsrn=np.array(datasaron)
    fd=np.array(datademung)
    fp=np.array(datapeking)
    fbb=np.array(databonangbarung)
    fbp=np.array(databonangpenerus)
    fsl=np.array(dataslenthem)
    fg=np.array(datagong)
    fk=np.array(datakendhang)
    imgmix=np.array(imaginer)
    
    numstep=np.size(ft,2)
    #resize 
    ft=np.resize(ft,[nsample,numstep])
    fsrn=np.resize(fsrn,[nsample,numstep])
    fd=np.resize(fd,[nsample,numstep])
    fp=np.resize(fp,[nsample,numstep])
    fbb=np.resize(fbb,[nsample,numstep])
    fbp=np.resize(fbp,[nsample,numstep])
    fsl=np.resize(fsl,[nsample,numstep])
    fg=np.resize(fg,[nsample,numstep])
    fk=np.resize(fk,[nsample,numstep])
    #i=np.resize(imgmix,[nsample,numstep])
    #imaginer_mix=i.astype(complex)
  
   
    return ft,fsrn,fd,fp,fbb,fbp,fsl,fg,fk

def invers_fourier(msaron,mdemung, mpeking,mbonangbarung,win,hop,numpad):
    i_saron=[]
    i_demung=[]
    i_peking=[]
    i_bonangbarung=[]
    ns=np.size(mdemung,0)
    nt=np.size(mdemung,1)
    numt=nt-(numpad*2)
    rsaron=np.resize(msaron,[ns,nt])
    rdemung=np.resize(mdemung,[ns,nt])
    rpeking=np.resize(mpeking,[ns,nt])
    rbonangbarung=np.resize(mbonangbarung,[ns,nt])
    
    psaron=rsaron[:,numpad:]
    psaron1=psaron[:,:numt]
    pdemung=rdemung[:,numpad:]
    pdemung1=pdemung[:,:numt]
    
    ppeking=rpeking[:,numpad:]
    ppeking1=ppeking[:,:numt]
    
    pbonangbarung=rbonangbarung[:,numpad:]
    pbonangbarung1=pbonangbarung[:,:numt]
    
    x_saron=psaron1.astype(complex)
    x_demung=pdemung1.astype(complex)
    #x_demung.imag=i
    x_peking=ppeking1.astype(complex)
    x_bonangbarung=pbonangbarung1.astype(complex)
    #x_peking.imag=i
    for i in range(ns):
        print("i=",i)
        invers_saron=tf.signal.irfft(x_saron[i:i+1,:]) 
        in_saron=invers_saron.numpy()
        transsaron=np.transpose(in_saron)
        i_saron.append(transsaron)    
    
        
        invers_demung=tf.signal.irfft(x_demung[i:i+1,:]) 
        in_demung=invers_demung.numpy()
        transdemung=np.transpose(in_demung)
        
        i_demung.append(transdemung)
        
        invers_peking=tf.signal.irfft(x_peking[i:i+1,:]) 
        in_peking=invers_peking.numpy()
        transpeking=np.transpose(in_peking)
        
        i_peking.append(transpeking)
        
        invers_bonangbarung=tf.signal.irfft(x_bonangbarung[i:i+1,:]) 
        in_bonangbarung=invers_bonangbarung.numpy()
        transbonangbarung=np.transpose(in_bonangbarung)
        
        i_bonangbarung.append(transbonangbarung)
    
    sr=np.array(i_saron)
    dm=np.array(i_demung)
    pk=np.array(i_peking)
    bb=np.array(i_bonangbarung)
    
    #resize
    hsaron=np.resize(sr,[ns,win])
    hdemung=np.resize(dm,[ns,win])
    hpeking=np.resize(pk,[ns, win])
    hbonangbarung=np.resize(bb,[ns, win])
   
    #susun jadi satu sample lagu
  
    raw_sample=np.floor((ns*win)/2)
    num_row=raw_sample.astype(int)
    rawsaron=np.zeros([1,num_row])
    rawdemung=np.zeros([1,num_row])
    rawpeking=np.zeros([1,num_row])
    rawbonangbarung=np.zeros([1,num_row])
    k=1
    rawsaron[:,0:hop]=hsaron[0:1,0:hop]
    rawdemung[:,0:hop]=hdemung[0:1,0:hop]
    rawpeking[:,0:hop]=hpeking[0:1,0:hop]
    rawbonangbarung[:,0:hop]=hbonangbarung[0:1,0:hop]
    while(k<(ns-2)):
        a=k*hop
        b=((k+2)*hop)+2
        print("k=",k)
        print("a=",a)
        print("b=",b)
        
        rawsaron[:,a:b]=hsaron[k:k+1,:]
        rawdemung[:,a:b]=hdemung[k:k+1,:]
        rawpeking[:,a:b]=hpeking[k:k+1,:]
        rawbonangbarung[:,a:b]=hbonangbarung[k:k+1,:]
        #rawd=np.transpose(rawdemung)
        # ks=str(k)
        # wavio.write("../result/demung"+ks+".wav", rawd, 44100, sampwidth=2)
        k=k+1
    #jadikan wav file
    rs=np.transpose(rawsaron)
    rd=np.transpose(rawdemung)
    rp=np.transpose(rawpeking)
    rbb=np.transpose(rawbonangbarung)
    wavio.write("../wav_result/saron_1ms_normal.wav", rs, 44100, sampwidth=2)
    wavio.write("../wav_result/demung_1ms_normal.wav", rd, 44100, sampwidth=2)
    wavio.write("../wav_result/peking_1ms_normal.wav", rp, 44100, sampwidth=2)
    wavio.write("../wav_result/bonangbarung_1ms_normal.wav", rbb, 44100, sampwidth=2)
    
    return i_demung,transdemung,i_demung,dm,hdemung,rawdemung,num_row,x_demung
def invers_mask(pred_saron,pred_demung, pred_peking,pred_bonangbarung,ftrain):
    msaron=np.multiply(pred_saron,ftrain)
    mdemung=np.multiply(pred_demung,ftrain)
    mpeking=np.multiply(pred_peking,ftrain)
    mbonangbarung=np.multiply(pred_bonangbarung,ftrain)
    
    return msaron,mdemung, mpeking,mbonangbarung

def softmask(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang):
    msaron=np.nan_to_num(fysaron/fxtrain)
    mdemung=np.nan_to_num(fydemung/fxtrain)
    mpeking=np.nan_to_num(fypeking/fxtrain)
    mbonangbarung=np.nan_to_num(fybonangbarung/fxtrain)
    mbonangpenerus=np.nan_to_num(fybonangpenerus/fxtrain)
    mslenthem=np.nan_to_num(fyslenthem/fxtrain)
    mgong=np.nan_to_num(fygong/fxtrain)
    mkendhang=np.nan_to_num(fykendhang/fxtrain)
    
    timestamp=np.size(msaron,1)
    numsample=np.size(msaron,0)
    numfeature=1
    ctrain=np.resize(fxtrain,[numsample,timestamp,numfeature])
    csaron1=np.resize(msaron,[numsample,timestamp,numfeature])
    cdemung1=np.resize(mdemung,[numsample,timestamp,numfeature])
    cpeking1=np.resize(mpeking,[numsample,timestamp,numfeature])
    cbonangbarung1=np.resize(mbonangbarung,[numsample,timestamp,numfeature])
    cbonangpenerus1=np.resize(mbonangpenerus,[numsample,timestamp,numfeature])
    cslenthem1=np.resize(mslenthem,[numsample,timestamp,numfeature])
    cgong1=np.resize(mgong,[numsample,timestamp,numfeature])
    ckendhang1=np.resize(mkendhang,[numsample,timestamp,numfeature])
    #====================ZERO PAD
    ctrain2,csaron2,cdemung2,cpeking2, cbonangbarung2, cbonangpenerus2, cslenthem2, cgong2, ckendhang2=zeropad(ctrain,csaron1,cdemung1,cpeking1, cbonangbarung1, cbonangpenerus1, cslenthem1, cgong1, ckendhang1,numfeature)
    #======================
    
    ts=np.size(ctrain2,1)
    return ctrain2, csaron2,cdemung2,cpeking2, cbonangbarung2, cbonangpenerus2, cslenthem2, cgong2, ckendhang2, ts,numfeature

def binarymask(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang,numpad):
    msaron=np.divide(np.abs(fysaron),np.abs(fxtrain))
    #csaron=np.around(msaron,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    msaron[np.isnan(msaron)] = 1
    # Replace all values over 1 with 1
    msaron[msaron > 1] = 1

    mdemung=np.divide(np.abs(fydemung),np.abs(fxtrain))
    #cdemung=np.around(mdemung,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mdemung[np.isnan(mdemung)] = 1
    # Replace all values over 1 with 1
    mdemung[mdemung > 1] = 1
    
    mpeking=np.divide(np.abs(fypeking),np.abs(fxtrain))
    #cpeking=np.around(mpeking,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mpeking[np.isnan(mpeking)] = 1
    # Replace all values over 1 with 1
    mpeking[mpeking > 1] = 1
    mbonangbarung=np.divide(np.abs(fybonangbarung),np.abs(fxtrain))
    #cbonangbarung=np.around(mbonangbarung,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mbonangbarung[np.isnan(mbonangbarung)] = 1
    # Replace all values over 1 with 1
    mbonangbarung[mbonangbarung > 1] = 1
    mbonangpenerus=np.divide(np.abs(fybonangpenerus),np.abs(fxtrain))
    #cbonangpenerus=np.around(mbonangpenerus,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mbonangpenerus[np.isnan(mbonangpenerus)] = 1
    # Replace all values over 1 with 1
    mbonangpenerus[mbonangpenerus> 1] = 1
    mslenthem=np.divide(np.abs(fyslenthem),np.abs(fxtrain))
    #mslenthem=np.around(mslenthem,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mslenthem[np.isnan(mslenthem)] = 1
    # Replace all values over 1 with 1
    mslenthem[mslenthem > 1] = 1
    mgong=np.divide(np.abs(fygong),np.abs(fxtrain))
    #cgong=np.around(mgong,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mgong[np.isnan(mgong)] = 1
    # Replace all values over 1 with 1
    mgong[mgong > 1] = 1
    mkendhang=np.divide(np.abs(fykendhang),np.abs(fxtrain))
    #ckendhang=np.around(mkendhang,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mkendhang[np.isnan(mkendhang)] = 1
    # Replace all values over 1 with 1
    mkendhang[mkendhang > 1] = 1
    
    timestamp=np.size(msaron,1)
    numsample=np.size(msaron,0)
    numfeature=1
    ctrain=np.resize(fxtrain,[numsample,timestamp,numfeature])
    csaron1=np.resize(msaron,[numsample,timestamp,numfeature])
    cdemung1=np.resize(mdemung,[numsample,timestamp,numfeature])
    cpeking1=np.resize(mpeking,[numsample,timestamp,numfeature])
    cbonangbarung1=np.resize(mbonangbarung,[numsample,timestamp,numfeature])
    cbonangpenerus1=np.resize(mbonangpenerus,[numsample,timestamp,numfeature])
    cslenthem1=np.resize(mslenthem,[numsample,timestamp,numfeature])
    cgong1=np.resize(mgong,[numsample,timestamp,numfeature])
    ckendhang1=np.resize(mkendhang,[numsample,timestamp,numfeature])
    #====================ZERO PAD
    ctrain2,csaron2,cdemung2,cpeking2, cbonangbarung2, cbonangpenerus2, cslenthem2, cgong2, ckendhang2=zeropad(ctrain,csaron1,cdemung1,cpeking1, cbonangbarung1, cbonangpenerus1, cslenthem1, cgong1, ckendhang1,numfeature,numpad)
    #======================
    
    ts=np.size(ctrain2,1)
    
    return ctrain2,csaron2,cdemung2,cpeking2, cbonangbarung2, cbonangpenerus2, cslenthem2, cgong2, ckendhang2, ts,numfeature

def aslibinarymask(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang):
    msaron=np.divide(np.abs(fysaron),np.abs(fxtrain))
    csaron=np.around(msaron,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    csaron[np.isnan(csaron)] = 1
    # Replace all values over 1 with 1
    csaron[csaron > 1] = 1

    mdemung=np.divide(np.abs(fydemung),np.abs(fxtrain))
    cdemung=np.around(mdemung,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    cdemung[np.isnan(cdemung)] = 1
    # Replace all values over 1 with 1
    cdemung[cdemung > 1] = 1
    
    mpeking=np.divide(np.abs(fypeking),np.abs(fxtrain))
    cpeking=np.around(mpeking,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    cpeking[np.isnan(cpeking)] = 1
    # Replace all values over 1 with 1
    cpeking[cpeking > 1] = 1
    mbonangbarung=np.divide(np.abs(fybonangbarung),np.abs(fxtrain))
    cbonangbarung=np.around(mbonangbarung,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    cbonangbarung[np.isnan(cbonangbarung)] = 1
    # Replace all values over 1 with 1
    cbonangbarung[cbonangbarung > 1] = 1
    mbonangpenerus=np.divide(np.abs(fybonangpenerus),np.abs(fxtrain))
    cbonangpenerus=np.around(mbonangpenerus,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    cbonangpenerus[np.isnan(cbonangpenerus)] = 1
    # Replace all values over 1 with 1
    cbonangpenerus[cbonangpenerus> 1] = 1
    mslenthem=np.divide(np.abs(fyslenthem),np.abs(fxtrain))
    cslenthem=np.around(mslenthem,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    cslenthem[np.isnan(cslenthem)] = 1
    # Replace all values over 1 with 1
    cslenthem[cslenthem > 1] = 1
    mgong=np.divide(np.abs(fygong),np.abs(fxtrain))
    cgong=np.around(mgong,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    cgong[np.isnan(cgong)] = 1
    # Replace all values over 1 with 1
    cgong[cgong > 1] = 1
    mkendhang=np.divide(np.abs(fykendhang),np.abs(fxtrain))
    ckendhang=np.around(mkendhang,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    ckendhang[np.isnan(ckendhang)] = 1
    # Replace all values over 1 with 1
    ckendhang[ckendhang > 1] = 1
    
    timestamp=np.size(msaron,1)
    numsample=np.size(msaron,0)
    numfeature=1
    ctrain=np.resize(fxtrain,[numsample,timestamp,numfeature])
    csaron1=np.resize(csaron,[numsample,timestamp,numfeature])
    cdemung1=np.resize(cdemung,[numsample,timestamp,numfeature])
    cpeking1=np.resize(cpeking,[numsample,timestamp,numfeature])
    cbonangbarung1=np.resize(cbonangbarung,[numsample,timestamp,numfeature])
    cbonangpenerus1=np.resize(cbonangpenerus,[numsample,timestamp,numfeature])
    cslenthem1=np.resize(cslenthem,[numsample,timestamp,numfeature])
    cgong1=np.resize(cgong,[numsample,timestamp,numfeature])
    ckendhang1=np.resize(ckendhang,[numsample,timestamp,numfeature])
    #====================ZERO PAD
    ctrain2,csaron2,cdemung2,cpeking2, cbonangbarung2, cbonangpenerus2, cslenthem2, cgong2, ckendhang2=zeropad(ctrain,csaron1,cdemung1,cpeking1, cbonangbarung1, cbonangpenerus1, cslenthem1, cgong1, ckendhang1,numfeature)
    #======================
    
    ts=np.size(ctrain2,1)
    
    return ctrain2,csaron2,cdemung2,cpeking2, cbonangbarung2, cbonangpenerus2, cslenthem2, cgong2, ckendhang2, ts,numfeature

def binarymask1(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang):
    msaron=np.abs(fysaron)/(np.abs(fxtrain)-np.abs(fysaron))
    #csaron=np.around(msaron,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    msaron[np.isnan(msaron)] = 1
    # Replace all values over 1 with 1
    msaron[msaron > 1] = 1

    mdemung=np.abs(fydemung)/(np.abs(fxtrain)-np.abs(fydemung))
    #cdemung=np.around(mdemung,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mdemung[np.isnan(mdemung)] = 1
    # Replace all values over 1 with 1
    mdemung[mdemung > 1] = 1
    
    mpeking=np.abs(fypeking)/(np.abs(fxtrain)-np.abs(fypeking))
    #cpeking=np.around(mpeking,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mpeking[np.isnan(mpeking)] = 1
    # Replace all values over 1 with 1
    mpeking[mpeking > 1] = 1
    mbonangbarung=np.abs(fybonangbarung)/(np.abs(fxtrain)-np.abs(fybonangbarung))
    #cbonangbarung=np.around(mbonangbarung,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mbonangbarung[np.isnan(mbonangbarung)] = 1
    # Replace all values over 1 with 1
    mbonangbarung[mbonangbarung > 1] = 1
    mbonangpenerus=np.abs(fybonangpenerus)/(np.abs(fxtrain)-np.abs(fybonangpenerus))
    #cbonangpenerus=np.around(mbonangpenerus,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mbonangpenerus[np.isnan(mbonangpenerus)] = 1
    # Replace all values over 1 with 1
    mbonangpenerus[mbonangpenerus> 1] = 1
    mslenthem=np.abs(fyslenthem)/(np.abs(fxtrain)-np.abs(fyslenthem))
    #mslenthem=np.around(mslenthem,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mslenthem[np.isnan(mslenthem)] = 1
    # Replace all values over 1 with 1
    mslenthem[mslenthem > 1] = 1
    mgong=np.abs(fygong)/(np.abs(fxtrain)-np.abs(fygong))
    #cgong=np.around(mgong,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mgong[np.isnan(mgong)] = 1
    # Replace all values over 1 with 1
    mgong[mgong > 1] = 1
    mkendhang=np.abs(fykendhang)/(np.abs(fxtrain)-np.abs(fykendhang))
    #ckendhang=np.around(mkendhang,0)
    # Convert all nan in mask to 1 (it shouldnt matter if this is 0 or 1)
    mkendhang[np.isnan(mkendhang)] = 1
    # Replace all values over 1 with 1
    mkendhang[mdemung > 1] = 1
    
    timestamp=np.size(msaron,1)
    numsample=np.size(msaron,0)
    numfeature=1
    ctrain=np.resize(fxtrain,[numsample,timestamp,numfeature])
    csaron1=np.resize(msaron,[numsample,timestamp,numfeature])
    cdemung1=np.resize(mdemung,[numsample,timestamp,numfeature])
    cpeking1=np.resize(mpeking,[numsample,timestamp,numfeature])
    cbonangbarung1=np.resize(mbonangbarung,[numsample,timestamp,numfeature])
    cbonangpenerus1=np.resize(mbonangpenerus,[numsample,timestamp,numfeature])
    cslenthem1=np.resize(mslenthem,[numsample,timestamp,numfeature])
    cgong1=np.resize(mgong,[numsample,timestamp,numfeature])
    ckendhang1=np.resize(mkendhang,[numsample,timestamp,numfeature])
    #====================ZERO PAD
    ctrain2,csaron2,cdemung2,cpeking2, cbonangbarung2, cbonangpenerus2, cslenthem2, cgong2, ckendhang2=zeropad(ctrain,csaron1,cdemung1,cpeking1, cbonangbarung1, cbonangpenerus1, cslenthem1, cgong1, ckendhang1,numfeature)
    #======================
    
    ts=np.size(ctrain2,1)
    
    return ctrain2,csaron2,cdemung2,cpeking2, cbonangbarung2, cbonangpenerus2, cslenthem2, cgong2, ckendhang2, ts,numfeature    

def zeropad(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang,numfeature,numpad):
    numsample=np.size(fxtrain,0)
       
    zero=np.zeros([numsample,numpad,numfeature])
   
    fxtrain=np.concatenate((zero,fxtrain,zero),axis=1)
    fysaron=np.concatenate((zero,fysaron,zero),axis=1)
    fydemung=np.concatenate((zero,fydemung,zero),axis=1)
    fypeking=np.concatenate((zero,fypeking,zero),axis=1)
    fybonangbarung=np.concatenate((zero,fybonangbarung,zero),axis=1)
    fybonangpenerus=np.concatenate((zero,fybonangpenerus,zero),axis=1)
    fyslenthem=np.concatenate((zero,fyslenthem,zero),axis=1)
    fygong=np.concatenate((zero,fygong,zero),axis=1)
    fykendhang=np.concatenate((zero,fykendhang,zero),axis=1)
    return fxtrain,fysaron,fydemung,fypeking,fybonangbarung,fybonangpenerus,fyslenthem,fygong,fykendhang

def Editplothasil(history_sgd,history_rmsprop,history_adam):
    # weights_path = os.getcwd() + "\\plot\\1fitur\\loss.png"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 14})
    # plt.figure(figsize=(8,7))
    # plt.xlim(0,100)
    # #plt.ylim(0.1,1)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # #plt.title('Accuracy of 2 Second Temporal Feature')
    
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.legend(['Training loss', 'Validation loss'], loc='center right')
    # plt.savefig(weights_path,dpi=400)
    # plt.show()
    
    #weights_path = os.getcwd() + "acc.jpg"
    green_patch = mpatches.Patch(color='green', label='Adam')
    
    orange_patch = mpatches.Patch(color='orange', label='RMSprop')
    purple_patch = mpatches.Patch(color='purple', label='SGD')
    train_patch = mlines.Line2D([], [], color='black', label='Training')
    valid_patch = mlines.Line2D([], [], color='black', label='Validation', linestyle='dashed')
    plt.figure(figsize=(8,7))
    plt.xlim(0,50)
    plt.ylim(0,1)
    plt.plot(history_adam.history['accuracy'], color="green",linewidth = 2)
    plt.plot(history_adam.history['val_accuracy'], color="green", linestyle='dashed', linewidth = 2)
    plt.plot(history_rmsprop.history['accuracy'], color="orange",linewidth = 2)
    plt.plot(history_rmsprop.history['val_accuracy'], color="orange", linestyle='dashed', linewidth = 2)
    plt.plot(history_sgd.history['accuracy'], color="purple",linewidth = 2)
    plt.plot(history_sgd.history['val_accuracy'], color="purple", linestyle='dashed', linewidth = 2)
    
    #plt.title('Accuracy of 2 Second Temporal Feature')
    
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(handles=[green_patch,orange_patch, purple_patch, train_patch,valid_patch], loc='center right')
    plt.savefig("acc.jpg",dpi=1000)
    plt.show()
    
    # weights_path = os.getcwd() + "\\plot\\1fitur\\1sprec_1sayap.png"
    # plt.figure(figsize=(8,7))
    # plt.xlim(0,100)
    # plt.ylim(0,1)
    # plt.plot(history.history['precision'])
    # plt.plot(history.history['val_precision'])
    # #plt.title('Precision of 2 Second Temporal Feature')
    # plt.ylabel('Precision')
    # plt.xlabel('Epochs')
    # plt.legend(['Training precision', 'Validation precision'], loc='center right')
    # plt.savefig(weights_path,dpi=400)
    # plt.show()

    # weights_path = os.getcwd() + "\\plot\\1fitur\\1srecall_1sayap.png"
    # plt.figure(figsize=(8,7))
    # plt.xlim(0,100)
    # plt.ylim(0,1)
    # plt.plot(history.history['recall'])
    # plt.plot(history.history['val_recall'])
    # #plt.title('Recall of 2 Second Temporal Feature')
    # plt.ylabel('Recall')
    # plt.xlabel('Epochs')
    # plt.legend(['Training recall', 'Validation recall'], loc='center right')
    # plt.savefig(weights_path,dpi=400)
    # plt.show()
    

def savefile(ftrain,csaron,cdemung,cpeking, cbonangbarung, cbonangpenerus, cslenthem, cgong, ckendhang,timestamp,feature, kls_saron,kls_demung, kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang):
    with h5py.File('../matrix/1ms/mat1ms.h5', 'w') as h5f:
        h5f.create_dataset('ftrain', data=ftrain)
        h5f.create_dataset('csaron', data=csaron)
        h5f.create_dataset('cdemung', data=cdemung)
        h5f.create_dataset('cpeking', data=cpeking)
        h5f.create_dataset('cbonangbarung', data=cbonangbarung)
        h5f.create_dataset('cbonangpenerus', data=cbonangpenerus)
        h5f.create_dataset('cslenthem', data=cslenthem)
        h5f.create_dataset('cgong', data=cgong)
        h5f.create_dataset('ckendhang', data=ckendhang)
        h5f.create_dataset('kls_saron', data=kls_saron)
        h5f.create_dataset('kls_demung', data=kls_demung)
        h5f.create_dataset('kls_peking', data=kls_peking)
        h5f.create_dataset('kls_bonangbarung', data=kls_bonangbarung)
        h5f.create_dataset('kls_bonangpenerus', data=kls_bonangpenerus)
        h5f.create_dataset('kls_slenthem', data=kls_slenthem)
        h5f.create_dataset('kls_gong', data=kls_gong)
        h5f.create_dataset('kls_kendhang', data=kls_kendhang)
    
    return 0

def loadh5file(path):
    with h5py.File(path, 'r') as h5f:
        ftrain = h5f['ftrain'][:]
        ftrain1=np.array(ftrain)
        csaron = h5f['csaron'][:]
        csaron1=np.array(csaron)
        cdemung = h5f['cdemung'][:]
        cdemung1=np.array(cdemung)
        cpeking = h5f['cpeking'][:]
        cpeking1=np.array(cpeking)
        cbonangbarung = h5f['cbonangbarung'][:]
        cbonangbarung1=np.array(cbonangbarung)
        cbonangpenerus = h5f['cbonangpenerus'][:]
        cbonangpenerus1=np.array(cbonangpenerus)
        cslenthem = h5f['cslenthem'][:]
        cslenthem1=np.array(cslenthem)
        cgong = h5f['cgong'][:]
        cgong1=np.array(cgong)
        ckendhang = h5f['ckendhang'][:]
        ckendhang1=np.array(ckendhang)
        kls_saron= h5f['kls_saron'][:]
        kls_saron1=np.array(kls_saron)
        kls_demung= h5f['kls_demung'][:]
        kls_demung1=np.array(kls_demung)
        kls_peking= h5f['kls_peking'][:]
        kls_peking1=np.array(kls_peking)
        kls_bonangbarung= h5f['kls_bonangbarung'][:]
        kls_bonangbarung1=np.array(kls_bonangbarung)
        
        kls_bonangpenerus= h5f['kls_bonangpenerus'][:]
        kls_bonangpenerus1=np.array(kls_bonangpenerus)
        
        kls_slenthem= h5f['kls_slenthem'][:]
        kls_slenthem1=np.array(kls_slenthem)
        
        kls_gong= h5f['kls_gong'][:]
        kls_gong1=np.array(kls_gong)
        
        kls_kendhang= h5f['kls_kendhang'][:]
        kls_kendhang1=np.array(kls_kendhang)
        
        timestamp=np.size(ckendhang1,1)
        feature=np.size(ckendhang1,2)
    
    return ftrain1, csaron1, cdemung1, cpeking1, cbonangbarung1, cbonangpenerus1, cslenthem1, cgong1, ckendhang1,timestamp,feature, kls_saron1, kls_demung1, kls_peking1, kls_bonangbarung1, kls_bonangpenerus1, kls_slenthem1, kls_gong1,kls_kendhang1
    