#coba pake beberapa model deep learning
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, MaxPooling1D, BatchNormalization, Dropout, UpSampling1D, Dense, Concatenate
from tensorflow.keras.optimizers import Adam, SGD, Adadelta,RMSprop,Nadam, Adagrad

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import regularizers
import tensorflow.compat.v1 as tf1
K=tf.keras
import numpy as np

kl=tf.keras.losses.KLDivergence()



import os


def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)

def Unet(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang,timestamp,feature, kls_saron,kls_demung, kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang):
    
    input_layer=Input(shape=(timestamp,feature))
    #encoder blok1
    cnn11=Conv1D(64, 3, padding='causal', activation='relu')(input_layer)
    b11=BatchNormalization()(cnn11)
    cnn12=Conv1D(64, 3, padding='causal', activation='relu')(b11)
    b12=BatchNormalization()(cnn12)
    max1=MaxPooling1D(2,strides=2)(b12)
    
    #encoder block 2
    cnn21=Conv1D(128, 3, padding='causal', activation='relu')(max1)
    b21=BatchNormalization()(cnn21)
    cnn22=Conv1D(128, 3, padding='causal', activation='relu')(b21)
    b22=BatchNormalization()(cnn22)
    max2=MaxPooling1D(2,strides=2)(b22)
    
    #encoder block 3
    cnn31=Conv1D(256, 3, padding='causal', activation='relu')(max2)
    b31=BatchNormalization()(cnn31)
    cnn32=Conv1D(256, 3, padding='causal', activation='relu')(b31)
    b32=BatchNormalization()(cnn32)
    max3=MaxPooling1D(2,strides=2)(b32)
    
    #encoder block 4
    cnn41=Conv1D(512, 3, padding='causal', activation='relu')(max3)
    b41=BatchNormalization()(cnn41)
    cnn42=Conv1D(512, 3, padding='causal', activation='relu')(b41)
    b42=BatchNormalization()(cnn42)
    max4=MaxPooling1D(2,strides=2)(b42)
    
    #encoder 5
    cnn51=Conv1D(1024, 3, padding='causal', activation='relu')(max4)
    b51=BatchNormalization()(cnn51)
    cnn52=Conv1D(1024, 3, padding='causal', activation='relu')(b51)
    b52=BatchNormalization()(cnn52)
    
    #latent
    
    up5=UpSampling1D(size=2)(b52)
    cnn53=Conv1D(512, 2, padding='causal', activation='relu')(up5)
    b53=BatchNormalization()(cnn53)
    
    #decoder
    #skip connection
    s1=Concatenate(axis=2, name='skipcon-1')([b42, b53])
    
    #decoder4
    d41=Conv1D(256, 3, padding='causal', activation='relu')(s1)
    db41=BatchNormalization()(d41)
    d42=Conv1D(256, 3, padding='causal', activation='relu')(db41)
    db42=BatchNormalization()(d42)
    
    up4=UpSampling1D(size=2)(db42)
    d43=Conv1D(128, 2, padding='causal', activation='relu')(up4)
    db43=BatchNormalization()(d43)
    
    
    #skip connection
    s2=Concatenate(axis=2)([max2, db43])
    
    d31=Conv1D(128, 3, padding='causal', activation='relu')(s2)
    db31=BatchNormalization()(d31)
    d32=Conv1D(128, 3, padding='causal', activation='relu')(db31)
    db32=BatchNormalization()(d32)
    
    up5=UpSampling1D(size=2)(db32)
    d33=Conv1D(64, 2, padding='causal', activation='relu')(up5)
    db33=BatchNormalization()(d33)
    
    s3=Concatenate(axis=2)([max1, db33])
    d21=Conv1D(64, 3, padding='causal', activation='relu')(s3)
    db21=BatchNormalization()(d21)
    d22=Conv1D(64, 3, padding='causal', activation='relu')(db21)
    db22=BatchNormalization()(d22)
    
    up6=UpSampling1D(size=2)(db22)
    d11=Conv1D(64, 2, padding='causal', activation='relu')(up6)
    db11=BatchNormalization()(d11)
    
    #output layer
    outsaron=Conv1D(1,1, activation="sigmoid", name='saron_separation')(db11)
    
    outdemung=Conv1D(1,1, activation="sigmoid", name='demung_separation')(db11)
    outpeking=Conv1D(1,1, activation="sigmoid", name='peking_separation')(db11)
    outbonangbarung=Conv1D(1,1, activation="sigmoid", name='bonangbarung_separation')(db11)
    outbonangpenerus=Conv1D(1,1, activation="sigmoid", name='bonangpenerus_separation')(db11)
    outslenthem=Conv1D(1,1, activation="sigmoid", name='slenthem_separation')(db11)
    outgong=Conv1D(1,1, activation="sigmoid", name='gong_separation')(db11)
    outkendhang=Conv1D(1,1, activation="sigmoid", name='kendhang_separation')(db11)
    
    #mlp
    #con=Concatenate(axis=2)([outsaron,outdemung,outpeking,outbonangbarung,outbonangpenerus,outslenthem,outslenthem, outgong, outkendhang])
    # h1=Dense(128, activation='sigmoid')(con)
    # h2=Dense(128, activation='sigmoid')(h1)
    f_saron=Flatten()(outsaron)
    f_demung=Flatten()(outdemung)
    f_peking=Flatten()(outpeking)
    f_bonangbarung=Flatten()(outbonangbarung)
    f_bonangpenerus=Flatten()(outbonangpenerus)
    f_slenthem=Flatten()(outslenthem)
    f_gong=Flatten()(outgong)
    f_kendhang=Flatten()(outkendhang)
    
    outkelas_saron=Dense(1, activation='sigmoid', name='class_saron')(f_saron)
    outkelas_demung=Dense(1, activation='sigmoid', name='class_demung')(f_demung)
    outkelas_peking=Dense(1, activation='sigmoid', name='class_peking')(f_peking)
    outkelas_bonangbarung=Dense(1, activation='sigmoid', name='class_bonangbarung')(f_bonangbarung)
    outkelas_bonangpenerus=Dense(1, activation='sigmoid', name='class_bonangpenerus')(f_bonangpenerus)
    outkelas_slenthem=Dense(1, activation='sigmoid', name='class_slenthem')(f_slenthem)
    outkelas_gong=Dense(1, activation='sigmoid', name='class_gong')(f_gong)
    outkelas_kendhang=Dense(1, activation='sigmoid', name='class_kendhang')(f_kendhang)
    #outkelas=Conv1D(1,1, activation="relu", name='class')(db22) #multi class
    model = Model(input_layer, [outsaron, outdemung, outpeking, outbonangbarung, outbonangpenerus, outslenthem, outgong, outkendhang, outkelas_saron, outkelas_demung, outkelas_peking, outkelas_bonangbarung, outkelas_bonangpenerus, outkelas_slenthem, outkelas_gong, outkelas_kendhang])
    model.summary()
    model.compile("Adam",loss={'saron_separation':'mean_squared_error','demung_separation':'mean_squared_error','peking_separation': 'mean_squared_error','bonangbarung_separation': 'mean_squared_error','bonangpenerus_separation': 'mean_squared_error','slenthem_separation': 'mean_squared_error', 'gong_separation': 'mean_squared_error','kendhang_separation': 'mean_squared_error','class_saron':'binary_crossentropy', 'class_demung':'binary_crossentropy','class_peking':'binary_crossentropy','class_bonangbarung':'binary_crossentropy','class_bonangpenerus':'binary_crossentropy','class_slenthem':'binary_crossentropy', 'class_gong':'binary_crossentropy', 'class_kendhang':'binary_crossentropy'}, metrics={'class_saron':'accuracy','class_demung':'accuracy','class_peking':'accuracy', 'class_bonangbarung':'accuracy','class_bonangpenerus':'accuracy', 'class_slenthem':'accuracy','class_gong':'accuracy','class_kendhang':'accuracy'})
    dot_img_file = '../model_save/unet_93ms_normal.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    csvlog=tf.keras.callbacks.CSVLogger('../log/unet_93ms_normal.csv')
    history=model.fit(fxtrain,[fysaron,fydemung,fypeking,fybonangbarung,fybonangpenerus, fyslenthem, fygong, fykendhang, kls_saron, kls_demung, kls_peking, kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang], validation_split=0.2, epochs=200, batch_size=128,callbacks=[csvlog], verbose=2)
    weight = '../save_model/unet_93ms_normal.h5'
    model.save(weight)
    
    pred=model.predict(fxtrain)
    
    
          
    return model,history,pred

def MH_Unet(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang,timestamp,feature, kls_saron,kls_demung, kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang):
    
    input_layer=Input(shape=(timestamp,feature))
    #encoder blok1
    cnn11=Conv1D(64, 3, padding='causal', activation='relu')(input_layer)
    b11=BatchNormalization()(cnn11)
    cnn12=Conv1D(64, 3, padding='causal', activation='relu')(b11)
    b12=BatchNormalization()(cnn12)
    max1=MaxPooling1D(2,strides=2)(b12)
    
    #encoder block 2
    cnn21=Conv1D(128, 3, padding='causal', activation='relu')(max1)
    b21=BatchNormalization()(cnn21)
    cnn22=Conv1D(128, 3, padding='causal', activation='relu')(b21)
    b22=BatchNormalization()(cnn22)
    max2=MaxPooling1D(2,strides=2)(b22)
    
    #encoder block 3
    cnn31=Conv1D(256, 3, padding='causal', activation='relu')(max2)
    b31=BatchNormalization()(cnn31)
    cnn32=Conv1D(256, 3, padding='causal', activation='relu')(b31)
    b32=BatchNormalization()(cnn32)
    max3=MaxPooling1D(2,strides=2)(b32)
    
    #encoder block 4
    cnn41=Conv1D(512, 3, padding='causal', activation='relu')(max3)
    b41=BatchNormalization()(cnn41)
    cnn42=Conv1D(512, 3, padding='causal', activation='relu')(b41)
    b42=BatchNormalization()(cnn42)
    max4=MaxPooling1D(2,strides=2)(b42)
    
    #encoder 5
    cnn51=Conv1D(1024, 3, padding='causal', activation='relu')(max4)
    b51=BatchNormalization()(cnn51)
    cnn52=Conv1D(1024, 3, padding='causal', activation='relu')(b51)
    b52=BatchNormalization()(cnn52)
    
    #latent saron
    
    up5=UpSampling1D(size=2, name="decoder saron")(b52)
    cnn53=Conv1D(512, 2, padding='causal', activation='relu')(up5)
    b53=BatchNormalization()(cnn53)
    
    #decoder
    #skip connection
    s1=Concatenate(axis=2, name='skipcon-1')([b42, b53])
    
    #decoder4
    d41=Conv1D(256, 3, padding='causal', activation='relu')(s1)
    db41=BatchNormalization()(d41)
    d42=Conv1D(256, 3, padding='causal', activation='relu')(db41)
    db42=BatchNormalization()(d42)
    
    up4=UpSampling1D(size=2)(db42)
    d43=Conv1D(128, 2, padding='causal', activation='relu')(up4)
    db43=BatchNormalization()(d43)
    
    
    #skip connection
    s2=Concatenate(axis=2)([max2, db43])
    
    d31=Conv1D(128, 3, padding='causal', activation='relu')(s2)
    db31=BatchNormalization()(d31)
    d32=Conv1D(128, 3, padding='causal', activation='relu')(db31)
    db32=BatchNormalization()(d32)
    
    up5=UpSampling1D(size=2)(db32)
    d33=Conv1D(64, 2, padding='causal', activation='relu')(up5)
    db33=BatchNormalization()(d33)
    
    s3=Concatenate(axis=2)([max1, db33])
    d21=Conv1D(64, 3, padding='causal', activation='relu')(s3)
    db21=BatchNormalization()(d21)
    d22=Conv1D(64, 3, padding='causal', activation='relu')(db21)
    db22=BatchNormalization()(d22)
    
    up6=UpSampling1D(size=2)(db22)
    d11=Conv1D(64, 2, padding='causal', activation='relu')(up6)
    db11=BatchNormalization()(d11)
    outsaron=Conv1D(1,1, activation="sigmoid", name='saron_separation')(db11)
    
    #latent demung
    
    up5_d=UpSampling1D(size=2, name="decoder saron")(b52)
    cnn53_d=Conv1D(512, 2, padding='causal', activation='relu')(up5_d)
    b53_d=BatchNormalization()(cnn53_d)
    
    #decoder
    #skip connection
    s1_d=Concatenate(axis=2, name='skipcon-1')([b42, b53_d])
    
    #decoder4
    d41_d=Conv1D(256, 3, padding='causal', activation='relu')(s1_d)
    db41_d=BatchNormalization()(d41_d)
    d42_d=Conv1D(256, 3, padding='causal', activation='relu')(db41_d)
    db42_d=BatchNormalization()(d42_d)
    
    up4_d=UpSampling1D(size=2)(db42_d)
    d43_d=Conv1D(128, 2, padding='causal', activation='relu')(up4_d)
    db43_d=BatchNormalization()(d43_d)
    
    
    #skip connection
    s2_d=Concatenate(axis=2)([max2, db43_d])
    
    d31_d=Conv1D(128, 3, padding='causal', activation='relu')(s2_d)
    db31_d=BatchNormalization()(d31_d)
    d32_d=Conv1D(128, 3, padding='causal', activation='relu')(db31_d)
    db32_d=BatchNormalization()(d32_d)
    
    up5_d=UpSampling1D(size=2)(db32_d)
    d33_d=Conv1D(64, 2, padding='causal', activation='relu')(up5_d)
    db33_d=BatchNormalization()(d33_d)
    
    s3_d=Concatenate(axis=2)([max1, db33_d])
    d21_d=Conv1D(64, 3, padding='causal', activation='relu')(s3_d)
    db21_d=BatchNormalization()(d21_d)
    d22_d=Conv1D(64, 3, padding='causal', activation='relu')(db21_d)
    db22_d=BatchNormalization()(d22_d)
    
    up6_d=UpSampling1D(size=2)(db22_d)
    d11_d=Conv1D(64, 2, padding='causal', activation='relu')(up6_d)
    db11_d=BatchNormalization()(d11_d)
    #output layer
    outdemung=Conv1D(1,1, activation="sigmoid", name='demung_separation')(db11_d)
   
    #latent peking
    up5_p=UpSampling1D(size=2, name="decoder saron")(b52)
    cnn53_p=Conv1D(512, 2, padding='causal', activation='relu')(up5_p)
    b53_p=BatchNormalization()(cnn53_p)
    
    #decoder
    #skip connection
    s1_p=Concatenate(axis=2, name='skipcon-1')([b42, b53_p])
    
    #decoder4
    d41_p=Conv1D(256, 3, padding='causal', activation='relu')(s1_p)
    db41_p=BatchNormalization()(d41_p)
    d42_p=Conv1D(256, 3, padding='causal', activation='relu')(db41_p)
    db42_p=BatchNormalization()(d42_p)
    
    up4_p=UpSampling1D(size=2)(db42_p)
    d43_p=Conv1D(128, 2, padding='causal', activation='relu')(up4_p)
    db43_p=BatchNormalization()(d43_p)
    
    
    #skip connection
    s2_p=Concatenate(axis=2)([max2, db43_p])
    
    d31_p=Conv1D(128, 3, padding='causal', activation='relu')(s2_p)
    db31_p=BatchNormalization()(d31_p)
    d32_p=Conv1D(128, 3, padding='causal', activation='relu')(db31_p)
    db32_p=BatchNormalization()(d32_p)
    
    up5_p=UpSampling1D(size=2)(db32_p)
    d33_p=Conv1D(64, 2, padding='causal', activation='relu')(up5_p)
    db33_p=BatchNormalization()(d33_p)
    
    s3_p=Concatenate(axis=2)([max1, db33_p])
    d21_p=Conv1D(64, 3, padding='causal', activation='relu')(s3_p)
    db21_p=BatchNormalization()(d21_p)
    d22_p=Conv1D(64, 3, padding='causal', activation='relu')(db21_p)
    db22_p=BatchNormalization()(d22_p)
    
    up6_p=UpSampling1D(size=2)(db22_p)
    d11_p=Conv1D(64, 2, padding='causal', activation='relu')(up6_p)
    db11_p=BatchNormalization()(d11_p)
    #output layer
    outpeking=Conv1D(1,1, activation="sigmoid", name='peking_separation')(db11_p)
    
    #latent bonangbarung
    up5_bb=UpSampling1D(size=2, name="decoder saron")(b52)
    cnn53_bb=Conv1D(512, 2, padding='causal', activation='relu')(up5_bb)
    b53_bb=BatchNormalization()(cnn53_bb)
    
    #decoder
    #skip connection
    s1_bb=Concatenate(axis=2, name='skipcon-1')([b42, b53_bb])
    
    #decoder4
    d41_bb=Conv1D(256, 3, padding='causal', activation='relu')(s1_bb)
    db41_bb=BatchNormalization()(d41_bb)
    d42_bb=Conv1D(256, 3, padding='causal', activation='relu')(db41_bb)
    db42_bb=BatchNormalization()(d42_bb)
    
    up4_bb=UpSampling1D(size=2)(db42_bb)
    d43_bb=Conv1D(128, 2, padding='causal', activation='relu')(up4_bb)
    db43_bb=BatchNormalization()(d43_bb)
    
    
    #skip connection
    s2_bb=Concatenate(axis=2)([max2, db43_bb])
    
    d31_bb=Conv1D(128, 3, padding='causal', activation='relu')(s2_bb)
    db31_bb=BatchNormalization()(d31_bb)
    d32_bb=Conv1D(128, 3, padding='causal', activation='relu')(db31_bb)
    db32_bb=BatchNormalization()(d32_bb)
    
    up5_bb=UpSampling1D(size=2)(db32_bb)
    d33_bb=Conv1D(64, 2, padding='causal', activation='relu')(up5_bb)
    db33_bb=BatchNormalization()(d33_bb)
    
    s3_bb=Concatenate(axis=2)([max1, db33_bb])
    d21_bb=Conv1D(64, 3, padding='causal', activation='relu')(s3_bb)
    db21_bb=BatchNormalization()(d21_bb)
    d22_bb=Conv1D(64, 3, padding='causal', activation='relu')(db21_bb)
    db22_bb=BatchNormalization()(d22_bb)
    
    up6_bb=UpSampling1D(size=2)(db22_bb)
    d11_bb=Conv1D(64, 2, padding='causal', activation='relu')(up6_bb)
    db11_bb=BatchNormalization()(d11_bb)
    #output layer
    outbonangbarung=Conv1D(1,1, activation="sigmoid", name='bonangbarung_separation')(db11_bb)
    
    #latent bonang penerus
    up5_bp=UpSampling1D(size=2, name="decoder saron")(b52)
    cnn53_bp=Conv1D(512, 2, padding='causal', activation='relu')(up5_bp)
    b53_bp=BatchNormalization()(cnn53_bp)
    
    #decoder
    #skip connection
    s1_bp=Concatenate(axis=2, name='skipcon-1')([b42, b53_bp])
    
    #decoder4
    d41_bp=Conv1D(256, 3, padding='causal', activation='relu')(s1_bp)
    db41_bp=BatchNormalization()(d41_bp)
    d42_bp=Conv1D(256, 3, padding='causal', activation='relu')(db41_bp)
    db42_bp=BatchNormalization()(d42_bp)
    
    up4_bp=UpSampling1D(size=2)(db42_bp)
    d43_bp=Conv1D(128, 2, padding='causal', activation='relu')(up4_bp)
    db43_bp=BatchNormalization()(d43_bp)
    
    
    #skip connection
    s2_bp=Concatenate(axis=2)([max2, db43_bp])
    
    d31_bp=Conv1D(128, 3, padding='causal', activation='relu')(s2_bp)
    db31_bp=BatchNormalization()(d31_bp)
    d32_bp=Conv1D(128, 3, padding='causal', activation='relu')(db31_bp)
    db32_bp=BatchNormalization()(d32_bp)
    
    up5_bp=UpSampling1D(size=2)(db32_bp)
    d33_bp=Conv1D(64, 2, padding='causal', activation='relu')(up5_bp)
    db33_bp=BatchNormalization()(d33_bp)
    
    s3_bp=Concatenate(axis=2)([max1, db33_bp])
    d21_bp=Conv1D(64, 3, padding='causal', activation='relu')(s3_bp)
    db21_bp=BatchNormalization()(d21_bp)
    d22_bp=Conv1D(64, 3, padding='causal', activation='relu')(db21_bp)
    db22_bp=BatchNormalization()(d22_bp)
    
    up6_bp=UpSampling1D(size=2)(db22_bp)
    d11_bp=Conv1D(64, 2, padding='causal', activation='relu')(up6_bp)
    db11_bp=BatchNormalization()(d11_bp)
    #output layer
    outbonangpenerus=Conv1D(1,1, activation="sigmoid", name='bonangpenerus_separation')(db11_bp)
    
    #latent slenthem
    up5_sl=UpSampling1D(size=2, name="decoder saron")(b52)
    cnn53_sl=Conv1D(512, 2, padding='causal', activation='relu')(up5_sl)
    b53_sl=BatchNormalization()(cnn53_sl)
    
    #decoder
    #skip connection
    s1_sl=Concatenate(axis=2, name='skipcon-1')([b42, b53_sl])
    
    #decoder4
    d41_sl=Conv1D(256, 3, padding='causal', activation='relu')(s1_sl)
    db41_sl=BatchNormalization()(d41_sl)
    d42_sl=Conv1D(256, 3, padding='causal', activation='relu')(db41_sl)
    db42_sl=BatchNormalization()(d42_sl)
    
    up4_sl=UpSampling1D(size=2)(db42_sl)
    d43_sl=Conv1D(128, 2, padding='causal', activation='relu')(up4_sl)
    db43_sl=BatchNormalization()(d43_sl)
    
    
    #skip connection
    s2_sl=Concatenate(axis=2)([max2, db43_sl])
    
    d31_sl=Conv1D(128, 3, padding='causal', activation='relu')(s2_sl)
    db31_sl=BatchNormalization()(d31_sl)
    d32_sl=Conv1D(128, 3, padding='causal', activation='relu')(db31_sl)
    db32_sl=BatchNormalization()(d32_sl)
    
    up5_sl=UpSampling1D(size=2)(db32_sl)
    d33_sl=Conv1D(64, 2, padding='causal', activation='relu')(up5_sl)
    db33_sl=BatchNormalization()(d33_sl)
    
    s3_sl=Concatenate(axis=2)([max1, db33_sl])
    d21_sl=Conv1D(64, 3, padding='causal', activation='relu')(s3_sl)
    db21_sl=BatchNormalization()(d21_sl)
    d22_sl=Conv1D(64, 3, padding='causal', activation='relu')(db21_sl)
    db22_sl=BatchNormalization()(d22_sl)
    
    up6_sl=UpSampling1D(size=2)(db22_sl)
    d11_sl=Conv1D(64, 2, padding='causal', activation='relu')(up6_sl)
    db11_sl=BatchNormalization()(d11_sl)
    #output layer
    outslenthem=Conv1D(1,1, activation="sigmoid", name='slenthem_separation')(db11_sl)
    
    #latent gong
    up5_g=UpSampling1D(size=2, name="decoder saron")(b52)
    cnn53_g=Conv1D(512, 2, padding='causal', activation='relu')(up5_g)
    b53_g=BatchNormalization()(cnn53_g)
    
    #decoder
    #skip connection
    s1_g=Concatenate(axis=2, name='skipcon-1')([b42, b53_g])
    
    #decoder4
    d41_g=Conv1D(256, 3, padding='causal', activation='relu')(s1_g)
    db41_g=BatchNormalization()(d41_g)
    d42_g=Conv1D(256, 3, padding='causal', activation='relu')(db41_g)
    db42_g=BatchNormalization()(d42_g)
    
    up4_g=UpSampling1D(size=2)(db42_g)
    d43_g=Conv1D(128, 2, padding='causal', activation='relu')(up4_g)
    db43_g=BatchNormalization()(d43_g)
    
    
    #skip connection
    s2_g=Concatenate(axis=2)([max2, db43_g])
    
    d31_g=Conv1D(128, 3, padding='causal', activation='relu')(s2_g)
    db31_g=BatchNormalization()(d31_g)
    d32_g=Conv1D(128, 3, padding='causal', activation='relu')(db31_g)
    db32_g=BatchNormalization()(d32_g)
    
    up5_g=UpSampling1D(size=2)(db32_g)
    d33_g=Conv1D(64, 2, padding='causal', activation='relu')(up5_g)
    db33_g=BatchNormalization()(d33_g)
    
    s3_g=Concatenate(axis=2)([max1, db33_g])
    d21_g=Conv1D(64, 3, padding='causal', activation='relu')(s3_g)
    db21_g=BatchNormalization()(d21_g)
    d22_g=Conv1D(64, 3, padding='causal', activation='relu')(db21_g)
    db22_g=BatchNormalization()(d22_g)
    
    up6_g=UpSampling1D(size=2)(db22_g)
    d11_g=Conv1D(64, 2, padding='causal', activation='relu')(up6_g)
    db11_g=BatchNormalization()(d11_g)
    #output layer
    outgong=Conv1D(1,1, activation="sigmoid", name='gong_separation')(db11_g)
    
    #latent kendhang
    up5_k=UpSampling1D(size=2, name="decoder saron")(b52)
    cnn53_k=Conv1D(512, 2, padding='causal', activation='relu')(up5_k)
    b53_k=BatchNormalization()(cnn53_k)
    
    #decoder
    #skip connection
    s1_k=Concatenate(axis=2, name='skipcon-1')([b42, b53_k])
    
    #decoder4
    d41_k=Conv1D(256, 3, padding='causal', activation='relu')(s1_k)
    db41_k=BatchNormalization()(d41_k)
    d42_k=Conv1D(256, 3, padding='causal', activation='relu')(db41_k)
    db42_k=BatchNormalization()(d42_k)
    
    up4_k=UpSampling1D(size=2)(db42_k)
    d43_k=Conv1D(128, 2, padding='causal', activation='relu')(up4_k)
    db43_k=BatchNormalization()(d43_k)
    
    
    #skip connection
    s2_k=Concatenate(axis=2)([max2, db43_k])
    
    d31_k=Conv1D(128, 3, padding='causal', activation='relu')(s2_k)
    db31_k=BatchNormalization()(d31_k)
    d32_k=Conv1D(128, 3, padding='causal', activation='relu')(db31_k)
    db32_k=BatchNormalization()(d32_k)
    
    up5_k=UpSampling1D(size=2)(db32_k)
    d33_k=Conv1D(64, 2, padding='causal', activation='relu')(up5_k)
    db33_k=BatchNormalization()(d33_k)
    
    s3_k=Concatenate(axis=2)([max1, db33_k])
    d21_k=Conv1D(64, 3, padding='causal', activation='relu')(s3_k)
    db21_k=BatchNormalization()(d21_k)
    d22_k=Conv1D(64, 3, padding='causal', activation='relu')(db21_k)
    db22_k=BatchNormalization()(d22_k)
    
    up6_k=UpSampling1D(size=2)(db22_k)
    d11_k=Conv1D(64, 2, padding='causal', activation='relu')(up6_k)
    db11_k=BatchNormalization()(d11_k)
    #output layer
    outkendhang=Conv1D(1,1, activation="sigmoid", name='kendhang_separation')(db11_k)
    
    #mlp
    #con=Concatenate(axis=2)([outsaron,outdemung,outpeking,outbonangbarung,outbonangpenerus,outslenthem,outslenthem, outgong, outkendhang])
    # h1=Dense(128, activation='sigmoid')(con)
    # h2=Dense(128, activation='sigmoid')(h1)
    f_saron=Flatten()(outsaron)
    f_demung=Flatten()(outdemung)
    f_peking=Flatten()(outpeking)
    f_bonangbarung=Flatten()(outbonangbarung)
    f_bonangpenerus=Flatten()(outbonangpenerus)
    f_slenthem=Flatten()(outslenthem)
    f_gong=Flatten()(outgong)
    f_kendhang=Flatten()(outkendhang)
    
    outkelas_saron=Dense(1, activation='sigmoid', name='class_saron')(f_saron)
    outkelas_demung=Dense(1, activation='sigmoid', name='class_demung')(f_demung)
    outkelas_peking=Dense(1, activation='sigmoid', name='class_peking')(f_peking)
    outkelas_bonangbarung=Dense(1, activation='sigmoid', name='class_bonangbarung')(f_bonangbarung)
    outkelas_bonangpenerus=Dense(1, activation='sigmoid', name='class_bonangpenerus')(f_bonangpenerus)
    outkelas_slenthem=Dense(1, activation='sigmoid', name='class_slenthem')(f_slenthem)
    outkelas_gong=Dense(1, activation='sigmoid', name='class_gong')(f_gong)
    outkelas_kendhang=Dense(1, activation='sigmoid', name='class_kendhang')(f_kendhang)
    #outkelas=Conv1D(1,1, activation="relu", name='class')(db22) #multi class
    model = Model(input_layer, [outsaron, outdemung, outpeking, outbonangbarung, outbonangpenerus, outslenthem, outgong, outkendhang, outkelas_saron, outkelas_demung, outkelas_peking, outkelas_bonangbarung, outkelas_bonangpenerus, outkelas_slenthem, outkelas_gong, outkelas_kendhang])
    model.summary()
    model.compile("Adam",loss={'saron_separation':'mean_squared_error','demung_separation':'mean_squared_error','peking_separation': 'mean_squared_error','bonangbarung_separation': 'mean_squared_error','bonangpenerus_separation': 'mean_squared_error','slenthem_separation': 'mean_squared_error', 'gong_separation': 'mean_squared_error','kendhang_separation': 'mean_squared_error','class_saron':'binary_crossentropy', 'class_demung':'binary_crossentropy','class_peking':'binary_crossentropy','class_bonangbarung':'binary_crossentropy','class_bonangpenerus':'binary_crossentropy','class_slenthem':'binary_crossentropy', 'class_gong':'binary_crossentropy', 'class_kendhang':'binary_crossentropy'}, metrics={'class_saron':'accuracy','class_demung':'accuracy','class_peking':'accuracy', 'class_bonangbarung':'accuracy','class_bonangpenerus':'accuracy', 'class_slenthem':'accuracy','class_gong':'accuracy','class_kendhang':'accuracy'})
    dot_img_file = '../model_save/MHunet_adam_mse_loss_binarymask_1ms.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    csvlog=tf.keras.callbacks.CSVLogger('../log/MHunet_adam_mse_loss_binarymask_1ms.csv')
    history=model.fit(fxtrain,[fysaron,fydemung,fypeking,fybonangbarung,fybonangpenerus, fyslenthem, fygong, fykendhang, kls_saron, kls_demung, kls_peking, kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang], validation_split=0.2, epochs=100, batch_size=20,callbacks=[csvlog])
    weight = '../model_save/MHunet_adam_mse_loss_binarymask_1ms.h5'
    model.save(weight)
    
    pred=model.predict(fxtrain)
    
    
          
    return model,history,pred
def Unet_softmask(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang,timestamp,feature, kls_saron,kls_demung, kls_peking,kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang):
    
    input_layer=Input(shape=(timestamp,feature))
    #encoder blok1
    cnn11=Conv1D(64, 3, padding='causal', activation='relu')(input_layer)
    b11=BatchNormalization()(cnn11)
    cnn12=Conv1D(64, 3, padding='causal', activation='relu')(b11)
    b12=BatchNormalization()(cnn12)
    max1=MaxPooling1D(2,strides=2)(b12)
    
    #encoder block 2
    cnn21=Conv1D(128, 3, padding='causal', activation='relu')(max1)
    b21=BatchNormalization()(cnn21)
    cnn22=Conv1D(128, 3, padding='causal', activation='relu')(b21)
    b22=BatchNormalization()(cnn22)
    max2=MaxPooling1D(2,strides=2)(b22)
    
    #encoder block 3
    cnn31=Conv1D(256, 3, padding='causal', activation='relu')(max2)
    b31=BatchNormalization()(cnn31)
    cnn32=Conv1D(256, 3, padding='causal', activation='relu')(b31)
    b32=BatchNormalization()(cnn32)
    max3=MaxPooling1D(2,strides=2)(b32)
    
    #encoder block 4
    cnn41=Conv1D(512, 3, padding='causal', activation='relu')(max3)
    b41=BatchNormalization()(cnn41)
    cnn42=Conv1D(512, 3, padding='causal', activation='relu')(b41)
    b42=BatchNormalization()(cnn42)
    max4=MaxPooling1D(2,strides=2)(b42)
    
    #encoder 5
    cnn51=Conv1D(1024, 3, padding='causal', activation='relu')(max4)
    b51=BatchNormalization()(cnn51)
    cnn52=Conv1D(1024, 3, padding='causal', activation='relu')(b51)
    b52=BatchNormalization()(cnn52)
    
    #latent saron
    
    up5=UpSampling1D(size=2)(b52)
    cnn53=Conv1D(512, 2, padding='causal', activation='relu')(up5)
    b53=BatchNormalization()(cnn53)
    
    #decoder
    #skip connection
    s1=Concatenate(axis=2, name='skipcon-1')([b42, b53])
    
    #decoder4
    d41=Conv1D(256, 3, padding='causal', activation='relu')(s1)
    db41=BatchNormalization()(d41)
    d42=Conv1D(256, 3, padding='causal', activation='relu')(db41)
    db42=BatchNormalization()(d42)
    
    up4=UpSampling1D(size=2)(db42)
    d43=Conv1D(128, 2, padding='causal', activation='relu')(up4)
    db43=BatchNormalization()(d43)
    
    
    #skip connection
    s2=Concatenate(axis=2)([max2, db43])
    
    d31=Conv1D(128, 3, padding='causal', activation='relu')(s2)
    db31=BatchNormalization()(d31)
    d32=Conv1D(128, 3, padding='causal', activation='relu')(db31)
    db32=BatchNormalization()(d32)
    
    up5=UpSampling1D(size=2)(db32)
    d33=Conv1D(64, 2, padding='causal', activation='relu')(up5)
    db33=BatchNormalization()(d33)
    
    s3=Concatenate(axis=2)([max1, db33])
    d21=Conv1D(64, 3, padding='causal', activation='relu')(s3)
    db21=BatchNormalization()(d21)
    d22=Conv1D(64, 3, padding='causal', activation='relu')(db21)
    db22=BatchNormalization()(d22)
    
    up6=UpSampling1D(size=2)(db22)
    d11=Conv1D(64, 2, padding='causal', activation='relu')(up6)
    db11=BatchNormalization()(d11)
    
    #output layer
    outsaron=Conv1D(1,1, activation="relu", name='saron_separation')(db11)
    
    outdemung=Conv1D(1,1, activation="relu", name='demung_separation')(db11)
    outpeking=Conv1D(1,1, activation="relu", name='peking_separation')(db11)
    outbonangbarung=Conv1D(1,1, activation="relu", name='bonangbarung_separation')(db11)
    outbonangpenerus=Conv1D(1,1, activation="relu", name='bonangpenerus_separation')(db11)
    outslenthem=Conv1D(1,1, activation="relu", name='slenthem_separation')(db11)
    outgong=Conv1D(1,1, activation="relu", name='gong_separation')(db11)
    outkendhang=Conv1D(1,1, activation="relu", name='kendhang_separation')(db11)
    
    #mlp
    #con=Concatenate(axis=2)([outsaron,outdemung,outpeking,outbonangbarung,outbonangpenerus,outslenthem,outslenthem, outgong, outkendhang])
    # h1=Dense(128, activation='sigmoid')(con)
    # h2=Dense(128, activation='sigmoid')(h1)
    f_saron=Flatten()(outsaron)
    f_demung=Flatten()(outdemung)
    f_peking=Flatten()(outpeking)
    f_bonangbarung=Flatten()(outbonangbarung)
    f_bonangpenerus=Flatten()(outbonangpenerus)
    f_slenthem=Flatten()(outslenthem)
    f_gong=Flatten()(outgong)
    f_kendhang=Flatten()(outkendhang)
    
    outkelas_saron=Dense(1, activation='sigmoid', name='class_saron')(f_saron)
    outkelas_demung=Dense(1, activation='sigmoid', name='class_demung')(f_demung)
    outkelas_peking=Dense(1, activation='sigmoid', name='class_peking')(f_peking)
    outkelas_bonangbarung=Dense(1, activation='sigmoid', name='class_bonangbarung')(f_bonangbarung)
    outkelas_bonangpenerus=Dense(1, activation='sigmoid', name='class_bonangpenerus')(f_bonangpenerus)
    outkelas_slenthem=Dense(1, activation='sigmoid', name='class_slenthem')(f_slenthem)
    outkelas_gong=Dense(1, activation='sigmoid', name='class_gong')(f_gong)
    outkelas_kendhang=Dense(1, activation='sigmoid', name='class_kendhang')(f_kendhang)
    #outkelas=Conv1D(1,1, activation="relu", name='class')(db22) #multi class
    model = Model(input_layer, [outsaron, outdemung, outpeking, outbonangbarung, outbonangpenerus, outslenthem, outgong, outkendhang, outkelas_saron, outkelas_demung, outkelas_peking, outkelas_bonangbarung, outkelas_bonangpenerus, outkelas_slenthem, outkelas_gong, outkelas_kendhang])
    model.summary()
    model.compile("Adam",loss={'saron_separation':'mean_squared_error','demung_separation':'mean_squared_error','peking_separation': 'mean_squared_error','bonangbarung_separation': 'mean_squared_error','bonangpenerus_separation': 'mean_squared_error','slenthem_separation': 'mean_squared_error', 'gong_separation': 'mean_squared_error','kendhang_separation': 'mean_squared_error','class_saron':'binary_crossentropy', 'class_demung':'binary_crossentropy','class_peking':'binary_crossentropy','class_bonangbarung':'binary_crossentropy','class_bonangpenerus':'binary_crossentropy','class_slenthem':'binary_crossentropy', 'class_gong':'binary_crossentropy', 'class_kendhang':'binary_crossentropy'}, metrics={'class_saron':'accuracy','class_demung':'accuracy','class_peking':'accuracy', 'class_bonangbarung':'accuracy','class_bonangpenerus':'accuracy', 'class_slenthem':'accuracy','class_gong':'accuracy','class_kendhang':'accuracy'})
    dot_img_file = '../model_save/unet_adam_mse_loss_softmask.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    csvlog=tf.keras.callbacks.CSVLogger('../log/unet_adam_mse_loss_softmask.csv')
    history=model.fit(fxtrain,[fysaron,fydemung,fypeking,fybonangbarung,fybonangpenerus, fyslenthem, fygong, fykendhang, kls_saron, kls_demung, kls_peking, kls_bonangbarung, kls_bonangpenerus, kls_slenthem, kls_gong, kls_kendhang], validation_split=0.2, epochs=100, batch_size=20,callbacks=[csvlog])
    weight = '../model_save/unet_adam_mse_loss_softmask.h5'
    model.save(weight)
    
    pred=model.predict(fxtrain)
    
    
          
    return model,history,pred

def Unet2(fxtrain,fydemung,fypeking,timestamp,feature, kelas):
    
    input_layer=Input(shape=(timestamp,feature))
    #encoder blok1
    cnn11=Conv1D(64, 3, padding='causal', activation='relu')(input_layer)
    b11=BatchNormalization()(cnn11)
    cnn12=Conv1D(64, 3, padding='causal', activation='relu')(b11)
    b12=BatchNormalization()(cnn12)
    max1=MaxPooling1D(2,strides=2)(b12)
    
    #encoder block 2
    cnn21=Conv1D(128, 3, padding='causal', activation='relu')(max1)
    b21=BatchNormalization()(cnn21)
    cnn22=Conv1D(128, 3, padding='causal', activation='relu')(b21)
    b22=BatchNormalization()(cnn22)
    max2=MaxPooling1D(2,strides=2)(b22)
    
    #encoder block 3
    cnn31=Conv1D(256, 3, padding='causal', activation='relu')(max2)
    b31=BatchNormalization()(cnn31)
    cnn32=Conv1D(256, 3, padding='causal', activation='relu')(b31)
    b32=BatchNormalization()(cnn32)
    max3=MaxPooling1D(2,strides=2)(b32)
    
    #encoder block 4
    cnn41=Conv1D(512, 3, padding='causal', activation='relu')(max3)
    b41=BatchNormalization()(cnn41)
    cnn42=Conv1D(512, 3, padding='causal', activation='relu')(b41)
    b42=BatchNormalization()(cnn42)
    max4=MaxPooling1D(2,strides=2)(b42)
    
    #encoder 5
    cnn51=Conv1D(1024, 3, padding='causal', activation='relu')(max4)
    b51=BatchNormalization()(cnn51)
    cnn52=Conv1D(1024, 3, padding='causal', activation='relu')(b51)
    b52=BatchNormalization()(cnn52)
    
    #latent saron
    
    up5=UpSampling1D(size=2)(b52)
    cnn53=Conv1D(512, 2, padding='causal', activation='relu')(up5)
    b53=BatchNormalization()(cnn53)
    
    #decoder
    #skip connection
    s1=Concatenate(axis=2, name='skipcon-1')([b42, b53])
    
    #decoder4
    d41=Conv1D(256, 3, padding='causal', activation='relu')(s1)
    db41=BatchNormalization()(d41)
    d42=Conv1D(256, 3, padding='causal', activation='relu')(db41)
    db42=BatchNormalization()(d42)
    
    up4=UpSampling1D(size=2)(db42)
    d43=Conv1D(128, 2, padding='causal', activation='relu')(up4)
    db43=BatchNormalization()(d43)
    
    
    #skip connection
    s2=Concatenate(axis=2)([max2, db43])
    
    d31=Conv1D(128, 3, padding='causal', activation='relu')(s2)
    db31=BatchNormalization()(d31)
    d32=Conv1D(128, 3, padding='causal', activation='relu')(db31)
    db32=BatchNormalization()(d32)
    
    up5=UpSampling1D(size=2)(db32)
    d33=Conv1D(64, 2, padding='causal', activation='relu')(up5)
    db33=BatchNormalization()(d33)
    
    s3=Concatenate(axis=2)([max1, db33])
    d21=Conv1D(64, 3, padding='causal', activation='relu')(s3)
    db21=BatchNormalization()(d21)
    d22=Conv1D(64, 3, padding='causal', activation='relu')(db21)
    db22=BatchNormalization()(d22)
    
    up6=UpSampling1D(size=2)(db22)
    d11=Conv1D(64, 2, padding='causal', activation='relu')(up6)
    db11=BatchNormalization()(d11)
    
    #output layer
    
    outdemung=Conv1D(1,1, activation="sigmoid", name='demung_separation')(db11)
    outpeking=Conv1D(1,1, activation="sigmoid", name='peking_separation')(db11)
    
    #mlp
    con=Concatenate(axis=2)([outdemung,outpeking])
    h1=Dense(128, activation='relu')(con)
    h2=Dense(128, activation='relu')(h1)
    f=Flatten()(h2)
    outkelas=Dense(8, activation='relu', name='class')(f)
    #outkelas=Conv1D(1,1, activation="relu", name='class')(db22) #multi class
    model = Model(input_layer, [outdemung, outpeking, outkelas])
    model.summary()
    model.compile("SGD",loss={'demung_separation':kl,'peking_separation': kl,'class':'binary_crossentropy'}, metrics={'class':'accuracy'})
    csvlog=tf.keras.callbacks.CSVLogger('../log/unet.csv')
    history=model.fit(fxtrain,[fydemung,fypeking, kelas], validation_split=0.2, epochs=100, batch_size=20,callbacks=[csvlog])
    weight = '../model_save/unet.h5'
    model.save(weight)
    pred=model.predict(fxtrain)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
   
          
    return model,history,pred


def BUnet(fxtrain,fysaron,fydemung,fypeking, fybonangbarung, fybonangpenerus, fyslenthem, fygong, fykendhang,timestamp,feature, kelas):
    
    input_layer=Input(shape=(timestamp,feature))
    #encoder blok1
    cnn11=Conv1D(64, 3, padding='causal', activation='relu')(input_layer)
    b11=BatchNormalization()(cnn11)
    cnn12=Conv1D(64, 3, padding='causal', activation='relu')(b11)
    b12=BatchNormalization()(cnn12)
    max1=MaxPooling1D(2,strides=2)(b12)
    
    #encoder block 2
    cnn21=Conv1D(128, 3, padding='causal', activation='relu')(max1)
    b21=BatchNormalization()(cnn21)
    cnn22=Conv1D(128, 3, padding='causal', activation='relu')(b21)
    b22=BatchNormalization()(cnn22)
    max2=MaxPooling1D(2,strides=2)(b22)
    
    #encoder block 3
    cnn31=Conv1D(256, 3, padding='causal', activation='relu')(max2)
    b31=BatchNormalization()(cnn31)
    cnn32=Conv1D(256, 3, padding='causal', activation='relu')(b31)
    b32=BatchNormalization()(cnn32)
    max3=MaxPooling1D(2,strides=2)(b32)
    
    #encoder block 4
    cnn41=Conv1D(512, 3, padding='causal', activation='relu')(max3)
    b41=BatchNormalization()(cnn41)
    cnn42=Conv1D(512, 3, padding='causal', activation='relu')(b41)
    b42=BatchNormalization()(cnn42)
    max4=MaxPooling1D(2,strides=2)(b42)
    
    #encoder 5
    cnn51=Conv1D(1024, 3, padding='causal', activation='relu')(max4)
    b51=BatchNormalization()(cnn51)
    cnn52=Conv1D(1024, 3, padding='causal', activation='relu')(b51)
    b52=BatchNormalization()(cnn52)
    
    #latent saron
    
    up5=UpSampling1D(size=2)(b52)
    cnn53=Conv1D(512, 2, padding='causal', activation='relu')(up5)
    b53=BatchNormalization()(cnn53)
    
    #decoder
    #skip connection
    s1=Concatenate(axis=2)([b42, b53])
    
    #decoder4
    d41=Conv1D(256, 3, padding='causal', activation='relu')(s1)
    db41=BatchNormalization()(d41)
    d42=Conv1D(256, 3, padding='causal', activation='relu')(db41)
    db42=BatchNormalization()(d42)
    
    up4=UpSampling1D(size=2)(db42)
    d43=Conv1D(128, 2, padding='causal', activation='relu')(up4)
    db43=BatchNormalization()(d43)
    
    
    #skip connection
    s2=Concatenate(axis=2)([max2, db43])
    
    d31=Conv1D(128, 3, padding='causal', activation='relu')(s2)
    db31=BatchNormalization()(d31)
    d32=Conv1D(128, 3, padding='causal', activation='relu')(db31)
    db32=BatchNormalization()(d32)
    
    up5=UpSampling1D(size=2)(db32)
    d33=Conv1D(64, 2, padding='causal', activation='relu')(up5)
    db33=BatchNormalization()(d33)
    
    s3=Concatenate(axis=2)([max1, db33])
    d21=Conv1D(64, 3, padding='causal', activation='relu')(s3)
    db21=BatchNormalization()(d21)
    d22=Conv1D(64, 3, padding='causal', activation='relu')(db21)
    db22=BatchNormalization()(d22)
    
    up6=UpSampling1D(size=2)(db22)
    d11=Conv1D(64, 2, padding='causal', activation='relu')(up6)
    db11=BatchNormalization()(d11)
    
    #output layer
    outsaron=Conv1D(1,1, activation="sigmoid", name='saron_separation')(db11)
    
    #latent demung
    dup5=UpSampling1D(size=2)(b52)
    dcnn53=Conv1D(512, 2, padding='causal', activation='relu')(dup5)
    db53=BatchNormalization()(dcnn53)
    
    #decoder
    #skip connection
    ds1=Concatenate(axis=2)([b42, db53])
    
    #decoder4
    dd41=Conv1D(256, 3, padding='causal', activation='relu')(ds1)
    ddb41=BatchNormalization()(dd41)
    dd42=Conv1D(256, 3, padding='causal', activation='relu')(ddb41)
    ddb42=BatchNormalization()(dd42)
    
    dup4=UpSampling1D(size=2)(ddb42)
    dd43=Conv1D(128, 2, padding='causal', activation='relu')(dup4)
    ddb43=BatchNormalization()(dd43)
    
    
    #skip connection
    ds2=Concatenate(axis=2)([max2, ddb43])
    
    dd31=Conv1D(128, 3, padding='causal', activation='relu')(ds2)
    ddb31=BatchNormalization()(dd31)
    dd32=Conv1D(128, 3, padding='causal', activation='relu')(ddb31)
    ddb32=BatchNormalization()(dd32)
    
    dup5=UpSampling1D(size=2)(ddb32)
    dd33=Conv1D(64, 2, padding='causal', activation='relu')(dup5)
    ddb33=BatchNormalization()(dd33)
    
    ds3=Concatenate(axis=2)([max1, ddb33])
    dd21=Conv1D(64, 3, padding='causal', activation='relu')(ds3)
    ddb21=BatchNormalization()(dd21)
    dd22=Conv1D(64, 3, padding='causal', activation='relu')(ddb21)
    ddb22=BatchNormalization()(dd22)
    
    dup6=UpSampling1D(size=2)(ddb22)
    dd11=Conv1D(64, 2, padding='causal', activation='relu')(dup6)
    ddb11=BatchNormalization()(dd11)
    
    
    outdemung=Conv1D(1,1, activation="sigmoid", name='demung_separation')(ddb11)
    
    #latent peking
    pup5=UpSampling1D(size=2)(b52)
    pcnn53=Conv1D(512, 2, padding='causal', activation='relu')(pup5)
    pb53=BatchNormalization()(pcnn53)
    
    #decoder
    #skip connection
    ps1=Concatenate(axis=2)([b42, pb53])
    
    #decoder4
    pd41=Conv1D(256, 3, padding='causal', activation='relu')(ps1)
    pdb41=BatchNormalization()(pd41)
    pd42=Conv1D(256, 3, padding='causal', activation='relu')(pdb41)
    pdb42=BatchNormalization()(pd42)
    
    pup4=UpSampling1D(size=2)(pdb42)
    pd43=Conv1D(128, 2, padding='causal', activation='relu')(pup4)
    pdb43=BatchNormalization()(pd43)
    
    
    #skip connection
    ps2=Concatenate(axis=2)([max2, pdb43])
    
    pd31=Conv1D(128, 3, padding='causal', activation='relu')(ps2)
    pdb31=BatchNormalization()(pd31)
    pd32=Conv1D(128, 3, padding='causal', activation='relu')(pdb31)
    pdb32=BatchNormalization()(pd32)
    
    pup5=UpSampling1D(size=2)(pdb32)
    pd33=Conv1D(64, 2, padding='causal', activation='relu')(pup5)
    pdb33=BatchNormalization()(pd33)
    
    ps3=Concatenate(axis=2)([max1, pdb33])
    pd21=Conv1D(64, 3, padding='causal', activation='relu')(ps3)
    pdb21=BatchNormalization()(pd21)
    pd22=Conv1D(64, 3, padding='causal', activation='relu')(pdb21)
    pdb22=BatchNormalization()(pd22)
    
    pup6=UpSampling1D(size=2)(pdb22)
    pd11=Conv1D(64, 2, padding='causal', activation='relu')(pup6)
    pdb11=BatchNormalization()(pd11)
    outpeking=Conv1D(1,1, activation="sigmoid", name='peking_separation')(pdb11)
    
    #latent bonang barung
    bbup5=UpSampling1D(size=2)(b52)
    bbcnn53=Conv1D(512, 2, padding='causal', activation='relu')(bbup5)
    bbb53=BatchNormalization()(bbcnn53)
    
    #decoder
    #skip connection
    bbs1=Concatenate(axis=2)([b42, bbb53])
    
    #decoder4
    bbd41=Conv1D(256, 3, padding='causal', activation='relu')(bbs1)
    bbdb41=BatchNormalization()(bbd41)
    bbd42=Conv1D(256, 3, padding='causal', activation='relu')(bbdb41)
    bbdb42=BatchNormalization()(bbd42)
    
    bbup4=UpSampling1D(size=2)(bbdb42)
    bbd43=Conv1D(128, 2, padding='causal', activation='relu')(bbup4)
    bbdb43=BatchNormalization()(bbd43)
    
    
    #skip connection
    bbs2=Concatenate(axis=2)([max2, bbdb43])
    
    bbd31=Conv1D(128, 3, padding='causal', activation='relu')(bbs2)
    bbdb31=BatchNormalization()(bbd31)
    bbd32=Conv1D(128, 3, padding='causal', activation='relu')(bbdb31)
    bbdb32=BatchNormalization()(bbd32)
    
    bbup5=UpSampling1D(size=2)(bbdb32)
    bbd33=Conv1D(64, 2, padding='causal', activation='relu')(bbup5)
    bbdb33=BatchNormalization()(bbd33)
    
    bbs3=Concatenate(axis=2)([max1, bbdb33])
    bbd21=Conv1D(64, 3, padding='causal', activation='relu')(bbs3)
    bbdb21=BatchNormalization()(bbd21)
    bbd22=Conv1D(64, 3, padding='causal', activation='relu')(bbdb21)
    bbdb22=BatchNormalization()(bbd22)
    
    bbup6=UpSampling1D(size=2)(bbdb22)
    bbd11=Conv1D(64, 2, padding='causal', activation='relu')(bbup6)
    bbdb11=BatchNormalization()(bbd11)
    
    outbonangbarung=Conv1D(1,1, activation="sigmoid",name='bonangbarung_separation')(bbdb11)
    
    #latent bonang penerus
    bpup5=UpSampling1D(size=2)(b52)
    bpcnn53=Conv1D(512, 2, padding='causal', activation='relu')(bpup5)
    bpb53=BatchNormalization()(bpcnn53)
    
    #decoder
    #skip connection", 
    bps1=Concatenate(axis=2)([b42, bpb53])
    
    #decoder4
    bpd41=Conv1D(256, 3, padding='causal', activation='relu')(bps1)
    bpdb41=BatchNormalization()(bpd41)
    bpd42=Conv1D(256, 3, padding='causal', activation='relu')(bpdb41)
    bpdb42=BatchNormalization()(bpd42)
    
    bpup4=UpSampling1D(size=2)(bpdb42)
    bpd43=Conv1D(128, 2, padding='causal', activation='relu')(bpup4)
    bpdb43=BatchNormalization()(bpd43)
    
    
    #skip connection
    bps2=Concatenate(axis=2)([max2, bpdb43])
    
    bpd31=Conv1D(128, 3, padding='causal', activation='relu')(bps2)
    bpdb31=BatchNormalization()(bpd31)
    bpd32=Conv1D(128, 3, padding='causal', activation='relu')(bpdb31)
    bpdb32=BatchNormalization()(bpd32)
    
    bpup5=UpSampling1D(size=2)(bpdb32)
    bpd33=Conv1D(64, 2, padding='causal', activation='relu')(bpup5)
    bpdb33=BatchNormalization()(bpd33)
    
    bps3=Concatenate(axis=2)([max1, bpdb33])
    bpd21=Conv1D(64, 3, padding='causal', activation='relu')(bps3)
    bpdb21=BatchNormalization()(bpd21)
    bpd22=Conv1D(64, 3, padding='causal', activation='relu')(bpdb21)
    bpdb22=BatchNormalization()(bpd22)
    
    bpup6=UpSampling1D(size=2)(bpdb22)
    bpd11=Conv1D(64, 2, padding='causal', activation='relu')(bpup6)
    bpdb11=BatchNormalization()(bpd11)
    
    outbonangpenerus=Conv1D(1,1, activation="sigmoid", name='bonangpenerus_separation')(bpdb11)
    
    #latent slenthem
    slup5=UpSampling1D(size=2)(b52)
    slcnn53=Conv1D(512, 2, padding='causal', activation='relu')(slup5)
    slb53=BatchNormalization()(slcnn53)
    
    #decoder
    #skip connection
    sls1=Concatenate(axis=2)([b42, slb53])
    
    #decoder4
    sld41=Conv1D(256, 3, padding='causal', activation='relu')(sls1)
    sldb41=BatchNormalization()(sld41)
    sld42=Conv1D(256, 3, padding='causal', activation='relu')(sldb41)
    sldb42=BatchNormalization()(sld42)
    
    slup4=UpSampling1D(size=2)(sldb42)
    sld43=Conv1D(128, 2, padding='causal', activation='relu')(slup4)
    sldb43=BatchNormalization()(sld43)
    
    
    #skip connection
    sls2=Concatenate(axis=2)([max2, sldb43])
    
    sld31=Conv1D(128, 3, padding='causal', activation='relu')(sls2)
    sldb31=BatchNormalization()(sld31)
    sld32=Conv1D(128, 3, padding='causal', activation='relu')(sldb31)
    sldb32=BatchNormalization()(sld32)
    
    slup5=UpSampling1D(size=2)(sldb32)
    sld33=Conv1D(64, 2, padding='causal', activation='relu')(slup5)
    sldb33=BatchNormalization()(sld33)
    
    sls3=Concatenate(axis=2)([max1, sldb33])
    sld21=Conv1D(64, 3, padding='causal', activation='relu')(sls3)
    sldb21=BatchNormalization()(sld21)
    sld22=Conv1D(64, 3, padding='causal', activation='relu')(sldb21)
    sldb22=BatchNormalization()(sld22)
    
    slup6=UpSampling1D(size=2)(sldb22)
    sld11=Conv1D(64, 2, padding='causal', activation='relu')(slup6)
    sldb11=BatchNormalization()(sld11)
    
    outslenthem=Conv1D(1,1, activation="sigmoid", name='slenthem_separation')(sldb11)
    
    #latent gong
    gup5=UpSampling1D(size=2)(b52)
    gcnn53=Conv1D(512, 2, padding='causal', activation='relu')(gup5)
    gb53=BatchNormalization()(gcnn53)
    
    #decoder
    #skip connection
    gs1=Concatenate(axis=2)([b42, gb53])
    
    #decoder4
    gd41=Conv1D(256, 3, padding='causal', activation='relu')(gs1)
    gdb41=BatchNormalization()(gd41)
    gd42=Conv1D(256, 3, padding='causal', activation='relu')(gdb41)
    gdb42=BatchNormalization()(gd42)
    
    gup4=UpSampling1D(size=2)(gdb42)
    gd43=Conv1D(128, 2, padding='causal', activation='relu')(gup4)
    gdb43=BatchNormalization()(gd43)
    
    
    #skip connection
    gs2=Concatenate(axis=2)([max2, gdb43])
    
    gd31=Conv1D(128, 3, padding='causal', activation='relu')(gs2)
    gdb31=BatchNormalization()(gd31)
    gd32=Conv1D(128, 3, padding='causal', activation='relu')(gdb31)
    gdb32=BatchNormalization()(gd32)
    
    gup5=UpSampling1D(size=2)(gdb32)
    gd33=Conv1D(64, 2, padding='causal', activation='relu')(gup5)
    gdb33=BatchNormalization()(gd33)
    
    gs3=Concatenate(axis=2)([max1, gdb33])
    gd21=Conv1D(64, 3, padding='causal', activation='relu')(gs3)
    gdb21=BatchNormalization()(gd21)
    gd22=Conv1D(64, 3, padding='causal', activation='relu')(gdb21)
    gdb22=BatchNormalization()(gd22)
    
    gup6=UpSampling1D(size=2)(gdb22)
    gd11=Conv1D(64, 2, padding='causal', activation='relu')(gup6)
    gdb11=BatchNormalization()(gd11)
    
    
    outgong=Conv1D(1,1, activation="sigmoid", name='gong_separation')(gdb11)
    
    #latent kendhang
    kup5=UpSampling1D(size=2)(b52)
    kcnn53=Conv1D(512, 2, padding='causal', activation='relu')(kup5)
    kb53=BatchNormalization()(kcnn53)
    
    #decoder
    #skip connection
    ks1=Concatenate(axis=2)([b42, kb53])
    
    #decoder4
    kd41=Conv1D(256, 3, padding='causal', activation='relu')(ks1)
    kdb41=BatchNormalization()(kd41)
    kd42=Conv1D(256, 3, padding='causal', activation='relu')(kdb41)
    kdb42=BatchNormalization()(kd42)
    
    kup4=UpSampling1D(size=2)(kdb42)
    kd43=Conv1D(128, 2, padding='causal', activation='relu')(kup4)
    kdb43=BatchNormalization()(kd43)
    
    
    #skip connection
    ks2=Concatenate(axis=2)([max2, kdb43])
    
    kd31=Conv1D(128, 3, padding='causal', activation='relu')(ks2)
    kdb31=BatchNormalization()(kd31)
    kd32=Conv1D(128, 3, padding='causal', activation='relu')(kdb31)
    kdb32=BatchNormalization()(kd32)
    
    kup5=UpSampling1D(size=2)(kdb32)
    kd33=Conv1D(64, 2, padding='causal', activation='relu')(kup5)
    kdb33=BatchNormalization()(kd33)
    
    ks3=Concatenate(axis=2)([max1, kdb33])
    kd21=Conv1D(64, 3, padding='causal', activation='relu')(ks3)
    kdb21=BatchNormalization()(kd21)
    kd22=Conv1D(64, 3, padding='causal', activation='relu')(kdb21)
    kdb22=BatchNormalization()(kd22)
    
    kup6=UpSampling1D(size=2)(kdb22)
    kd11=Conv1D(64, 2, padding='causal', activation='relu')(kup6)
    kdb11=BatchNormalization()(kd11)
    
    outkendhang=Conv1D(1,1, activation="sigmoid", name='kendhang_separation')(kdb11)
    
    
    #mlp
    # con=Concatenate(axis=2)([outsaron,outdemung,outpeking,outbonangbarung,outbonangpenerus,outslenthem,outslenthem, outgong, outkendhang])
    # h1=Dense(128, activation='relu')(con)
    # h2=Dense(128, activation='relu')(h1)
    # f=Flatten()(h2)
    # d=Dropout(0.3)(f)
    # outkelas=Dense(8, activation='relu', name='class')(d)
    # #outkelas=Conv1D(1,1, activation="relu", name='class')(db22) #multi class
    model = Model(input_layer, [outsaron, outdemung, outpeking, outbonangbarung, outbonangpenerus, outslenthem, outgong, outkendhang])
    model.summary()
    model.compile("SGD",loss={'saron_separation':'mean_squared_error','demung_separation':'mean_squared_error','peking_separation': 'mean_squared_error','bonangbarung_separation': 'mean_squared_error','bonangpenerus_separation': 'mean_squared_error','slenthem_separation': 'mean_squared_error', 'gong_separation': 'mean_squared_error','kendhang_separation': 'mean_squared_error'})
    csvlog=tf.keras.callbacks.CSVLogger('../log/unet.csv')
    history=model.fit(fxtrain,[fysaron,fydemung,fypeking,fybonangbarung,fybonangpenerus, fyslenthem, fygong, fykendhang], validation_split=0.2, epochs=100, batch_size=20,callbacks=[csvlog])
    weight = '../model_save/unet.h5'
    model.save(weight)
    pred=model.predict(fxtrain)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
   
          
    return model,history,pred


