# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:52:03 2025

@author: l
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import fftconvolve, convolve
from scipy.signal import stft, welch

#%% audio
fs=1000
fs_audio_c, wav_data_c = sio.wavfile.read('la cucaracha.wav')
N_c = len(wav_data_c)

f_welch, Pxx = sig.welch(wav_data_c, fs_audio_c, nfft=N_c, window='hann', nperseg=N_c//16, axis=0)

# STFT
f, t_stft, Zxx = stft(wav_data_c, fs=fs_audio_c, nperseg=256*2) # Ampliamos la ventana temporal 
    
# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)
    
# Señal
ax1.plot(wav_data_c)
ax1.set_title("Audio: 'La cucaracha'")
ax1.set_ylabel("Amplitud")
    
# Subplot 2: Welch
ax2.semilogy(f_welch, Pxx)
ax2.set_title("Estimación espectral por Welch")
ax2.set_ylabel("PSD [V²/Hz]")
ax2.set_xlabel("Frecuencia [Hz]")
    
    # Espectrograma
pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
ax3.set_title("STFT (Espectrograma)")
ax3.set_ylabel("Frecuencia [Hz]")
ax3.set_xlabel("Tiempo [s]")
ax3.set_ylim(0, 3000)
    
# Colorbar en eje externo
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")
    
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()


#%% ECG 

fs_ecg = 1000 # Hz

#### ECG CON RUIDO ###

#Visualizamos lo que contiene el archivo
#io.whosmat('ECG_TP4.mat')

#Guardamos los datos del archivo en un diccionario de Python: enlistamos
mat_struct = sio.loadmat('./ECG_TP4.mat')

#Extraemos los datos de la señal y los metemos en un vector.
ecg_one_lead = mat_struct['ecg_lead'].flatten()
ecg_one_lead= ecg_one_lead[5000:12000]
N1 = len(ecg_one_lead)
f_ecg_, Pxx_ecg_ = sig.welch(ecg_one_lead, fs_ecg, nfft=N1, window='hann', nperseg=N1//6, axis=0)

# STFT
f, t_stft, Zxx = stft(ecg_one_lead, fs=fs_ecg, nperseg=1000) # Ampliamos la ventana temporal 
    
# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)
    
# Señal
ax1.plot(ecg_one_lead)
ax1.set_title("ECG")
ax1.set_ylabel("Amplitud")
    
# Subplot 2: Welch
ax2.semilogy(f_ecg_, Pxx_ecg_)
ax2.set_title("Estimación espectral por Welch")
ax2.set_ylabel("PSD [V²/Hz]")
ax2.set_xlabel("Frecuencia [Hz]")
    
# Espectrograma
pcm = ax3.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
ax3.set_title("STFT (Espectrograma)")
ax3.set_ylabel("Frecuencia [Hz]")
ax3.set_xlabel("Tiempo [s]")
ax3.set_ylim(0, 30)

# Colorbar en eje externo
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")
    
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()





