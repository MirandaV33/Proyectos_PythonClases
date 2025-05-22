# -*- coding: utf-8 -*-
"""
Created on Wed May 21 21:06:36 2025

@author: l
"""


import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

#%% Diseño de filtro digital 

#%% Señal ECG

# Tenemos componentes de baja frecuencias que generan interferencia!
# Por sistemas electricos tenemos interferencia de 50Hz 
# Vamos a buscar que nuestro filtro sea un  PASA BANDA, atenuando las freuencias bajas y las altas. 

fs_ecg = 1000 # Hz

### ECG SIN RUIDO ###
ecg_one_lead = np.load('ecg_sin_ruido.npy')
N2 = len(ecg_one_lead)
ecg_one_lead= ecg_one_lead / np.std(ecg_one_lead)

f_ecg, Pxx_ecg = sig.welch(ecg_one_lead, fs_ecg, nfft=N2, window='hann', nperseg=N2//6, axis=0)
print(np.max(Pxx_ecg))


