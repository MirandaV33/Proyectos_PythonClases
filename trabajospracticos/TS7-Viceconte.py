# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 20:26:27 2025

@author: l
"""


#%% Librerias

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
#Plantilla
from pytc2.sistemas_lineales import plot_plantilla

#%% Definicion de funciones 
        
def plot_regions(ecg_signal, ecg_signal_filt, regs_interes, demora, label='ECG filtrado', crear_figura=True):
    
    cant_muestras = len(ecg_signal)
    for ii in regs_interes:
        zoom_region = np.arange(max(0, ii[0]), min(cant_muestras, ii[1]), dtype='uint')

        if crear_figura:
            plt.figure(figsize=(16, 8), facecolor='w', edgecolor='k')

        plt.plot(zoom_region, ecg_signal[zoom_region], label='ECG', linewidth=2)
        plt.plot(zoom_region, ecg_signal_filt[zoom_region + demora], label=label)
        
        plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
        plt.ylabel('Adimensional')
        plt.xlabel('Muestras (#)')
        
        axes_hdl = plt.gca()
        axes_hdl.legend()
        axes_hdl.set_yticks(())

        if crear_figura:
            plt.show()


#%% Importamos los datos del ECG

#Guardamos los datos del archivo en un diccionario de Python: enlistamos
mat_struct = sio.loadmat('./ECG_TP4.mat')

#Extraemos los datos de la señal y los metemos en un vector.
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N1 = len(ecg_one_lead)

# Normalizo la señal: 
ecg_one_lead = ecg_one_lead / np.std(ecg_one_lead)

# Defino variables que voy a analizar dentro del filtro.

# #Complejo de ondas QRS normal
qrs_pattern1 = mat_struct['qrs_pattern1'].flatten()
# Latido normal 
heartbeat_pattern1= mat_struct['heartbeat_pattern1'].flatten()
# Latido de origen ventricular
heartbeat_pattern2 = mat_struct['heartbeat_pattern2'].flatten()
# Vector con las localizaciones (en # de muestras) donde ocurren los latidos
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
    'Señal con detecciones QRS': qrs_detections  # para marcar qrs_detections
}

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
axs = axs.flatten()  # para usar un solo índice: axs[0], axs[1], ...

for idx, (titulo, senial) in enumerate(seniales.items()):
    axs[idx].plot(senial, label=titulo)
    axs[idx].set_title(f'{titulo}')
    axs[idx].set_xlabel('Muestras')
    axs[idx].set_ylabel('Amplitud [V]')
    axs[idx].grid(True)
    axs[idx].legend()

plt.tight_layout()
plt.show()

#%% Diseño de la Plantilla del filtro --> Como queremos que se comporten nuestros filtros
# Para el ecg, tenemos ruido de baja frecuencia (ruido de linea y red electrica) y ruido de alta frecuencia. 
# Se diseña un filtro pasa banda. 

# Parámetros de la plantilla
fs = 1000 # Hz
nyq_frec = fs/2
ripple = 1 # dB
attenuation = 40 # dB

fpass = np.array([1.0, 35.0]) #Hz

#Atenuamos ruido de base de linea (0.1) y ruido de red electrica (50) junto con el de alta frecuencia
fstop = np.array([.1, 50.]) # Hz

plt.figure(figsize=(10, 6))
plt.title('Plantilla de Diseño de Filtro Pasabanda para ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.legend()
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
ax = plt.gca()
plt.show()

#%%  Diseños de filtros

#%% IIR 
aprox_name = ['butter', 'cheby1']
filtros_i= []
# Crear figura con subplots 2x2
fig, axs = plt.subplots(1, 2, sharey=True)
axs = axs.flatten()  # Para indexar con un solo índice

# Eje de frecuencia
w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250))
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True)) / nyq_frec * np.pi

# Recorrer cada filtro
for idx, ftype in enumerate(aprox_name):
    mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=ftype, output='sos', fs=fs)
    filtros_i.append(mi_sos)
    w, hh = sig.sosfreqz(mi_sos, worN=w_rad)
    
    
    axs[idx].plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(hh) + 1e-15), label=ftype)
    axs[idx].set_title(f'Aproximación: {ftype}')
    axs[idx].set_xlabel('Frecuencia [Hz]')
    axs[idx].set_ylabel('Amplitud [dB]')
    axs[idx].grid(True)
    axs[idx].legend()

    # Llamar a tu función para la plantilla si acepta un eje
    plt.sca(axs[idx])
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop,
                   attenuation=attenuation, fs=fs)

# Título general y ajuste
plt.suptitle('Filtros IIR con diferentes métodos de aproximación', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% FIR
freq = [0, fstop[0], fpass[0], fpass[1], fpass[1]+1, nyq_frec]
gain = [0, 0, 1, 1, 0, 0] 

# Ventanas
Filtro_Ventana = sig.firwin2(numtaps=2505, freq=freq, gain=gain, window='hamming', fs=fs)

# Cuadrados mínimos--> Requerimiento asimetric en frecuencia
# Opcion 1: Concatenar un pasa altos y un pasa bajos
# Opcion 2: cambiar la banda de paso! Hacerla simetrica 

numtaps_ls= 1505
Filtro_LS = sig.firls(numtaps_ls, freq, gain, fs=fs)

filtros_fir = {
    'Método de ventanas': Filtro_Ventana,
    'Método de cuadrados mínimos': Filtro_LS
}

# Visualización 
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axs = axs.flatten()

for idx, (nombre, filtro) in enumerate(filtros_fir.items()):
    w, h = sig.freqz(filtro, worN=8000, fs=fs)
    axs[idx].plot(w, 20 * np.log10(np.abs(h) + 1e-15), label=nombre)
    axs[idx].set_title(nombre)
    axs[idx].set_xlabel('Frecuencia [Hz]')
    axs[idx].set_ylabel('Amplitud [dB]')
    axs[idx].set_ylim(-100,10)  # Mostrará desde -100 dB hasta +30 dB
    axs[idx].grid(True)
    axs[idx].legend()
    # Plantilla en cada subplot
    plt.sca(axs[idx])
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple,
                   fstop=fstop, attenuation=attenuation, fs=fs)

plt.suptitle('Filtros FIR', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% Analisis de filtros 

#%% IR

# Filtramos las señales
ecgs_filtrados = {}  
nombres_filtros = ['Butter', 'Cheby I']

for nombre, sos in zip(nombres_filtros, filtros_i):
    ecg_filt = sig.sosfiltfilt(sos, ecg_one_lead)  # filtrar sin distorsión de fase
    ecgs_filtrados[nombre] = ecg_filt
    plt.figure() 
    plt.plot(ecg_one_lead, label= 'Señal sin filtrar')
    plt.plot(ecg_filt, label='Señal filtrada ')
    plt.title(f'ECG: Electrocardiograma filtrado por -{nombre}')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    plt.legend()
    plt.show()

# Regiones sin ruido (convertidas de minutos a muestras)
regs_con_ruido = (
    np.array([5, 5.2]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

demora = 0

for nombre, ecg_filt in ecgs_filtrados.items():
    fig, axs = plt.subplots(1, len(regs_con_ruido), figsize=(18, 5), sharey=True)
    axs = axs.flatten()
    
    for i, reg in enumerate(regs_con_ruido):
        plt.sca(axs[i])
        plot_regions(ecg_one_lead, ecg_filt, [reg], demora, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)

    plt.suptitle(f'{nombre} - Regiones SIN ruido', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Regiones con ruido
regs_sin_ruido = (
    [4000, 5500],
    [10000, 11000],
)
demora2=80


for nombre, ecg_filt in ecgs_filtrados.items():
    fig, axs = plt.subplots(1, len(regs_sin_ruido), figsize=(12, 5), sharey=True)
    axs = axs.flatten()
    
    for i, reg in enumerate(regs_sin_ruido):
        plt.sca(axs[i])
        plot_regions(ecg_one_lead, ecg_filt, [reg], demora2, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región ruido {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)

    plt.suptitle(f'{nombre} - Regiones CON ruido', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    
#%% FIR 

ecgs_filtrados_fir = {}
nombres_filtros = ['Ventanas', 'Cuadrados Minimos']

for nombre, filtro in filtros_fir.items():
    ecg_filt = sig.sosfilt(filtros_fir, ecg_one_lead)  # FIR usa lfilter
    ecgs_filtrados_fir[nombre] = ecg_filt
    plt.figure() 
    plt.plot(ecg_one_lead, label= 'Señal sin filtrar')
    plt.plot(ecg_filt, label='Señal filtrada')
    plt.title(f'ECG: Electrocardiograma filtrado por -{nombre}')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    plt.legend()
    plt.show()

# Regiones con ruido (convertidas de minutos a muestras)
regs_con_ruido = (
    np.array([5, 5.2]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

demora = 0

for nombre, ecg_filt in ecgs_filtrados_fir.items():
    fig, axs = plt.subplots(1, len(regs_con_ruido), figsize=(18, 5), sharey=True)
    axs = axs.flatten()
    
    for i, reg in enumerate(regs_con_ruido):
        plt.sca(axs[i])
        plot_regions(ecg_one_lead, ecg_filt, [reg], demora, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)

    plt.suptitle(f'{nombre} - Regiones con ruido', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# Regiones con ruido
regs__sin_ruido = (
    [4000, 5500],
    [10000, 11000],
)
demora2=0


for nombre, ecg_filt in ecgs_filtrados_fir.items():
    fig, axs = plt.subplots(1, len(regs__sin_ruido), figsize=(12, 5), sharey=True)
    axs = axs.flatten()
    
    for i, reg in enumerate(regs__sin_ruido):
        plt.sca(axs[i])
        plot_regions(ecg_one_lead, ecg_filt, [reg], demora2, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región ruido {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)

    plt.suptitle(f'{nombre} - Regiones sin ruido', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    
