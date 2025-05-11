# -*- coding: utf-8 -*-
"""
Created on Wed May  7 20:14:28 2025

@author: l
"""


import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write
from scipy.io.wavfile import read

#Extraigo datos de la señales
fs_audio_c, wav_data_c = sio.wavfile.read('la cucaracha.wav')
fs_audio_s, wav_data_s = sio.wavfile.read('silbido.wav')
fs_audio_p, wav_data_p = sio.wavfile.read('prueba psd.wav')

N = len(wav_data_c)
N = len(wav_data_s)
N = len(wav_data_p)

#Normalizo en potencia: maximo entorno a -40dB
wavv_data= wav_data_c/np.std(wav_data_c)
wavv_data= wav_data_s/np.std(wav_data_s)
wavv_data= wav_data_p/np.std(wav_data_p)

#Normalizo 2: Respecto al maximo, esto es para que todos tengan su pico al 0dB. El ancho de banda nos juega en contra para comparar espectros de diferentes frecuencias
wavvv_data_c=wav_data_c/np.max(wav_data_c)
wavvv_data_s=wav_data_s/np.max(wav_data_s)
wavvv_data_p=wav_data_p/np.max(wav_data_p)

#%% Verfiicamos parseval en todas las señales

#"La cucaracha"
ft_wav_c = np.fft.fft(wavvv_data_c)
ft_Wav_c = np.abs(ft_wav_c)**2
parseval_c= np.mean(ft_Wav_c)

# "Silbido"
ft_wav_s = np.fft.fft(wavvv_data_s)
ft_Wav_s = np.abs(ft_wav_s)**2
parseva_sl= np.mean(ft_Wav_s)

# "Prueba" 
ft_wav_p = np.fft.fft(wavvv_data_p)
ft_Wav_p = np.abs(ft_wav_p)**2
parseval_p= np.mean(ft_Wav_p)

#%% Graficamos las señales 
plt.figure(1)
plt.plot(wavvv_data_c)
plt.title("Audio.wav: La cucaracha")
plt.legend()


plt.figure(2)
plt.plot(wavvv_data_s)
plt.title("Audio.wav: Silbido")
plt.legend()


plt.figure(3)
plt.plot(wavvv_data_p)
plt.title("Audio.wav: Prueba")
plt.legend()

#%% Estimacion por Welch

f,Pxx=sig.welch(wavvv_data_c,fs_audio_c,nfft=N,window='hann', nperseg= N//6,axis=0)
plt.figure(4)
plt.plot(f, 10* np.log10(2*np.abs(Pxx)**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density')
plt.show()


#%% Estimacion por Blackman Tukey 

# def blackman_tukey(x,  M = None):    
    
#     # N = len(x)
#     x_z = x.shape
    
#     N = np.max(x_z)
    
#     if M is None:
#         M = N//5
    
#     r_len = 2*M-1

#     # hay que aplanar los arrays por np.correlate.
#     # usaremos el modo same que simplifica el tratamiento
#     # de la autocorr
#     xx = x.ravel()[:r_len];

#     r = np.correlate(xx, xx, mode='same') / r_len

#     Px = np.abs(np.fft.fft(r * sig.windows.blackman(r_len), n = N) )

#     Px = Px.reshape(x_z)

#     return Px;

# Px = blackman_tukey(wav_data, M=N//10)

# # Eje de frecuencia
# f_bt = np.linspace(0, fs_audio, len(Px), endpoint=False)

# bfrec= f_bt <= fs_audio / 2

# plt.figure(3)
# plt.plot(f_bt, 10*np.log10(2*np.abs(Px[bfrec])**2))
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('PSD [dB]')
# plt.title('Estimación de PSD - Blackman-Tukey')
# plt.grid()
# plt.tight_layout()
# plt.show()




# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)