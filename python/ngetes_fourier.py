
#BUKTIKAN ANATARA MEMISAHKAN RIIL DAN IMAGINER DAN NILAI ABOSULT. MANA YANG LEBIH MIRIP SIGNAL ASLI
from baca import *
import tensorflow as tf

import numpy as np


from scipy.io.wavfile import write

combine_fft=[]
path='../data/dummy'
sr, chanel1,  length=bacasignal(path)
c1=np.array(chanel1)
awal=0
gabung=np.zeros([4410,1])
combine=[]
com_gabung=[]
for i in range(10):
    detik01=4410
    print("awal",awal)
    akhir=awal+detik01
    print("akhir",akhir)
    c_potong=c1[:,awal:akhir]
    fft_x1=tf.signal.rfft(c_potong)
    x1 = tf.keras.backend.abs(fft_x1)

    imag=tf.math.imag(fft_x1)
    real=tf.math.real(fft_x1)
    complx=tf.dtypes.complex(real, imag)
   # with tf.compat.v1.Session() as sess:
    r_fft= fft_x1.numpy()  
    r_abs= x1.numpy()
    img=imag.numpy()
    riil=real.numpy()
    h_complex=complx.numpy()
    combine_fft.append(r_abs)
    #========invers dari value real dan imaginerx
    #conver r_abs to complex
    potong_complex=r_abs.astype(complex)
    invers_x1=tf.signal.irfft(h_complex) 
    
    #invers dari fourier yang diabsolutkan
    invers_abs=tf.signal.irfft(potong_complex)
    
    with tf.compat.v1.Session() as sess:
        inv_c1=invers_x1.numpy()
        inv_abs=invers_abs.numpy()
    tf.keras.backend.clear_session()
    hasil_c1=np.transpose(inv_c1)
    #hasil_abs=np.transpose(invers_abs)
    detik2=np.transpose(c_potong)
    hsl=np.transpose(invers_abs)
    combine.append(detik2)
    com_gabung.append(hsl)
    #KESIMPULAN RIIL DAN IMAGINER LEBIH MIRIP SIGNAL ASLI SAAT DIKONVERSIKAN DARI INVERS FOURIER
    ii=str(i)
    write('detik'+ii+'.wav', sr, detik2)
    write('hasil_detik'+ii+'.wav', sr, hsl)
    #write('gabung'+ii+'.wav', sr, gabung)
    
    awal=akhir
    
hasil_combine=np.array(combine)
hsl_gabung=np.array(com_gabung)
re_combine=np.reshape(hasil_combine,[44100,1])
re_gabung=np.reshape(hsl_gabung,[44100,1])
write('combine.wav', sr, re_combine)
write('hasilinversall.wav', sr, re_gabung)