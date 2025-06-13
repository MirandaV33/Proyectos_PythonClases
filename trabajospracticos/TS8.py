# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 17:59:38 2025

@author: l
"""

#%% Librerias
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
#Plantilla
from pytc2.sistemas_lineales import plot_plantilla

#%% Importacion de datos

#%% Importamos los datos del ECG

mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N1 = len(ecg_one_lead)

# Normalizo la señal: 
ecg_one_lead = ecg_one_lead / np.std(ecg_one_lead)

# Defino variables que voy a analizar dentro del filtro.
qrs_pattern1 = mat_struct['qrs_pattern1'].flatten()
heartbeat_pattern1= mat_struct['heartbeat_pattern1'].flatten()
heartbeat_pattern2 = mat_struct['heartbeat_pattern2'].flatten()
qrs_detections = mat_struct['qrs_detections'].flatten()

# Visualizacion de la señal en tiempo 
plt.figure() 
plt.plot(ecg_one_lead, label= 'Señal de ECG completa')
plt.title("ECG: Electrocardiograma con ruido")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')
plt.grid(True)

seniales = {
    'QRS (latido normal)': qrs_pattern1,
    'Latido normal completo': heartbeat_pattern1,
    'Latido ventricular': heartbeat_pattern2,
    'Señal con detecciones QRS': qrs_detections
}

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
axs = axs.flatten()

for idx, (titulo, senial) in enumerate(seniales.items()):
    axs[idx].plot(senial, label=titulo)
    axs[idx].set_title(f'{titulo}')
    axs[idx].set_xlabel('Muestras')
    axs[idx].set_ylabel('Amplitud [V]')
    axs[idx].grid(True)
    axs[idx].legend()

plt.tight_layout()
plt.show()

#%% Diseño de filtros

