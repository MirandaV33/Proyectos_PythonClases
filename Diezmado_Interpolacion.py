# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 20:27:32 2025

@author: l
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla

#%% Datos a simular
aprox_name = 'butter'

fs = 1000 # Hz
M=100 #Bajamos el ancho de banda DIEZ veces pasamos de 500Hz a 50Hz (Muy justo)
nyq_frec= fs/4
fpass = fs/M #Hz
ripple = -1.0 # dB
attenuation = 40 # dB

#%% Cargo los datos del ECG

#Guardamos los datos del archivo en un diccionario de Python: enlistamos
mat_struct = sio.loadmat('./ECG_TP4.mat')

#Extraemos los datos de la señal y los metemos en un vector.
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N1 = len(ecg_one_lead)
# Normalizo la señal 
ecg_one_lead_r = ecg_one_lead / np.std(ecg_one_lead)


#%% 1. Diezmado: LLevar la señal a un ancho de banda reducido. Estimacion de movimiento de linea de base


# ecg_diezmado=ecg_one_lead_r[::M]

#  #Grafico SIN fltrar para ver como afecta el diezmado 
# t_original= np.arange(len(ecg_one_lead_r))/fs
# t_diezmado= t_original[::M]

# #Visualizacion de resultados
# plt.figure() 
# plt.plot(t_original, ecg_one_lead_r, label= 'Señal')
# plt.plot(t_diezmado, ecg_diezmado, label='Señal diezmada')
# plt.title("ECG: Electrocardiograma con ruido")
# plt.xlabel('Tiempo')
# plt.ylabel('Amplitud [V]')

#Con 100 se rompe! Ya no es tan parecido a la señal original

#Entonces filtramos y DESPUES diezmamos



ecg_diezmado=ecg_filtrada[::M]

 #Grafico SIN fltrar para ver como afecta el diezmado 
t_original= np.arange(len(ecg_one_lead_r))/fs
t_diezmado= t_original[::M]

#Visualizacion de resultados
plt.figure() 
plt.plot(t_original, ecg_one_lead_r, label= 'Señal')
plt.plot(t_diezmado, ecg_diezmado, label='Señal diezmada')
plt.title("ECG: Electrocardiograma con ruido")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')


#%% 2. Interpolacion: Cambio de muestreo sencillo (audio)




#%% 3. Diezmar el ECG: ancho de banda mas reducido que se pueda 128Hz. 
