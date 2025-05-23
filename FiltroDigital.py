# -*- coding: utf-8 -*-
"""
Created on Wed May 21 21:06:36 2025

@author: l
"""

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla

#%% Diseño de filtro digital 

#%% Señal ECG

# Tenemos componentes de baja frecuencias que generan interferencia!
# Por sistemas electricos tenemos interferencia de 50Hz 
# Vamos a buscar que nuestro filtro sea un  PASA BANDA, atenuando las freuencias bajas y las altas. 
### Graficos de salida: SOS 

#%% Definicion de funciones

#%% Datos a simular
aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip' cower

fs = 1000 # Hz
nyq_frec= fs/2
fpass = np.array([1.0, 35.0]) #Hz
ripple = 1.0 # dB
fstop = np.array([.1, 50.]) # Hz
attenuation = 40 # dB

#%% Diseño del filtro
mi_sos=sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output= 'sos', fs=fs)
###!!! 14 pares de ceros y polos, orden 28

#%% Analisis de filtro

npoints = 1000

w, hh = sig.sosfreqz(mi_sos, worN=npoints)

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad)

plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+1e-15), label='mi_sos')

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')


ax = plt.gca()
# ax.set_xlim([0, 1])
# ax.set_ylim([-60, 1])ç

plot_plantilla(filter_type = "bandpass" , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()


#### ECG CON RUIDO ###

#Guardamos los datos del archivo en un diccionario de Python: enlistamos
mat_struct = sio.loadmat('./ECG_TP4.mat')

#Extraemos los datos de la señal y los metemos en un vector.
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N1 = len(ecg_one_lead)

# Normalizo la señal 
ecg_one_lead_r = ecg_one_lead / np.std(ecg_one_lead)

#%% Filtro

ecg_filt= sig.sosfiltfilt(mi_sos, ecg_one_lead_r)

# %%Visualizacion de resultados 

###SEÑAL TEMPORAL###

#Sin filtrar
plt.figure() 
plt.plot(ecg_one_lead_r, label= 'Señal sin filtrar')
plt.plot(ecg_filt, label='Señal filtrada')
plt.title("ECG: Electrocardiograma con ruido")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')

#IMPORTANTE:en el dominio temporal podemos observar como se comporta la envolvente! Que en el espectro es dificil de apreciar
# Este tipo de filtro, altera la morfologia de un ECG porque tiene transcisiones muy abruptas. Tenemos que bajar el Q, habria que analizar la respuesta al impulso y transcicion. Influye directamente sobre la efectividad de la banda de paso. Redefinir la plantilla. 
#Al hacer filt filt, estamos filtrando el doble de atenuacion 40-->80dB
#El rippple esta relacionado al maximo de la señal, por lo que pasa 1-->2 dB
# Filt--> Distorsion de fase 

plt.show()