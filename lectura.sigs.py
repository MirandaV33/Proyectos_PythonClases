#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:55:30 2023

@author: mariano
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write
from scipy.io.wavfile import read


def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

##################
# Lectura de ECG #
##################

fs_ecg = 1000 # Hz

# para listar las variables que hay en el archivo
#io.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N = len(ecg_one_lead)

hb_1 = vertical_flaten(mat_struct['heartbeat_pattern1'])
hb_2 = vertical_flaten(mat_struct['heartbeat_pattern2'])

plt.figure()
plt.plot(ecg_one_lead[5000:12000])
plt.title("ECG: Electrocardiograma")
plt.legend()

plt.figure()
plt.plot(hb_1)
plt.title("Latidos tipicos: Normal")
plt.legend()
plt.show

plt.figure()
plt.title("Latidos tipicos: Ectópico") 
#Conduccion INVERSA a la normal: Pico mayor de mnor intensidad y tiempo de recuperacion. Uno cada tres,arritmia. Trigeminismo.
plt.plot(hb_2)
plt.legend()
plt.show

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

# Cargar el archivo CSV como un array de NumPy
ppg = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)  # Omitir la cabecera si existe

plt.figure()
plt.plot(ppg)
plt.title("PPG: ")

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
# fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')
# fs_audio, wav_data = sio.wavfile.read('silbido.wav')

plt.figure()
plt.plot(wav_data)

# si quieren oirlo, tienen que tener el siguiente módulo instalado
# pip install sounddevice
# import sounddevice as sd
# sd.play(wav_data, fs_audio)

