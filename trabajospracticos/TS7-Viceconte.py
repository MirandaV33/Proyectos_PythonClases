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

# Concatenar ambos filtros
def filtro_pasabanda(signal, filtro_alto, filtro_bajo):
    salida1 = np.convolve(signal, filtro_alto, mode='same')
    salida2 = np.convolve(salida1, filtro_bajo, mode='same')
    return salida2


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

#%% Plantilla del filtro

fs = 1000 # Hz
nyq_frec = fs/2
ripple = 1 # dB
attenuation = 40 # dB
fpass = np.array([1.0, 35.0]) #Hz
fstop = np.array([.1, 50.]) # Hz

plt.figure(figsize=(10, 6))
plt.title('Plantilla de Diseño de Filtro Pasabanda para ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.legend()
plt.show()

#%% Diseño filtros IIR

aprox_name = ['butter', 'cheby1']
filtros_i = {}
fig, axs = plt.subplots(1, 2, sharey=True)
axs = axs.flatten()

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250))
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True)) / nyq_frec * np.pi

for idx, ftype in enumerate(aprox_name):
    mi_sos = sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=ftype, output='sos', fs=fs)
    filtros_i[ftype.capitalize()] = mi_sos
    w, hh = sig.sosfreqz(mi_sos, worN=w_rad)

    axs[idx].plot(w / np.pi * fs / 2, 20 * np.log10(np.abs(hh) + 1e-15), label=ftype)
    axs[idx].set_title(f'Aproximación: {ftype}')
    axs[idx].set_xlabel('Frecuencia [Hz]')
    axs[idx].set_ylabel('Amplitud [dB]')
    axs[idx].grid(True)
    axs[idx].legend()

    plt.sca(axs[idx])
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)

plt.suptitle('Filtros IIR con diferentes métodos de aproximación', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%% Diseño FIR

freq = [0, fstop[0], fpass[0], fpass[1], fstop[1], nyq_frec]
gain = [0, 0, 1, 1, 0, 0] 

Filtro_Ventana = sig.firwin2(numtaps=2505, freq=freq, gain=gain, window='hamming', fs=fs)

numtaps_ls= 1505
Filtro_LS = sig.firls(numtaps_ls, freq, gain, fs=fs)

filtros_fir = {
    'Método de ventanas': Filtro_Ventana,
    'Método de cuadrados mínimos': Filtro_LS
}

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axs = axs.flatten()

for idx, (nombre, filtro) in enumerate(filtros_fir.items()):
    w, h = sig.freqz(filtro, worN=8000, fs=fs)
    axs[idx].plot(w, 20 * np.log10(np.abs(h) + 1e-15), label=nombre)
    axs[idx].set_title(nombre)
    axs[idx].set_xlabel('Frecuencia [Hz]')
    axs[idx].set_ylabel('Amplitud [dB]')
    axs[idx].set_ylim(-100,10)
    axs[idx].grid(True)
    axs[idx].legend()
    plt.sca(axs[idx])
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)

plt.suptitle('Filtros FIR', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#Restriccion de banda de paso 
freq = [0, fstop[0], fpass[0], fpass[1], fpass[1]+1, nyq_frec]
gain = [0, 0, 1, 1, 0, 0] 

Filtro_Ventana2 = sig.firwin2(numtaps=2505, freq=freq, gain=gain, window='hamming', fs=fs)

numtaps_ls= 1505
Filtro_LS = sig.firls(numtaps_ls, freq, gain, fs=fs)

filtros_fir = {
    'Método de ventanas': Filtro_Ventana2,
    'Método de cuadrados mínimos': Filtro_LS
}

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
axs = axs.flatten()

for idx, (nombre, filtro) in enumerate(filtros_fir.items()):
    w, h = sig.freqz(filtro, worN=8000, fs=fs)
    axs[idx].plot(w, 20 * np.log10(np.abs(h) + 1e-15), label=nombre)
    axs[idx].set_title(nombre)
    axs[idx].set_xlabel('Frecuencia [Hz]')
    axs[idx].set_ylabel('Amplitud [dB]')
    axs[idx].set_ylim(-100,10)
    axs[idx].grid(True)
    axs[idx].legend()
    plt.sca(axs[idx])
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)

plt.suptitle('Filtros FIR por simetrizacion', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#Diseño de filtros concatenados--Ventanas
filtro_pasalto = sig.firwin(numtaps=501, cutoff=fpass[0], fs=fs, pass_zero=False, window='hamming')
filtro_pasbajo = sig.firwin(numtaps=501, cutoff=fpass[1], fs=fs, pass_zero=True, window='hamming')

# Aplicar filtro pasa banda
signal_filtrada = filtro_pasabanda(ecg_one_lead, filtro_pasalto, filtro_pasbajo)

# Mostrar respuesta en frecuencia del filtro combinado
filtro_pbanda = np.convolve(filtro_pasalto, filtro_pasbajo)

w, h = sig.freqz(filtro_pbanda, worN=8000, fs=fs)

plt.figure(figsize=(10,4))
plt.plot(w, 20*np.log10(np.abs(h)+1e-15))
plt.title('Filtro pasa banda por ventanas (concatenacion')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.ylim([-100, 10])
plt.legend()
plt.grid(True)
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()

# Diseño con mínimos cuadrados

# PASA ALTOS: atenúa <0.1 Hz, pasa >1 Hz
freq_pa = [0, fstop[0], fpass[0], nyq_frec]
gain_pa = [0, 0, 1, 1]
numtaps=1501
filtro_pasalto = sig.firls(numtaps, freq_pa, gain_pa, fs=fs)

# PASA BAJOS: pasa <35 Hz, atenúa >50 Hz
freq_pb = [0, fpass[1], fstop[1], nyq_frec]
gain_pb = [1, 1, 0, 0]
numtaps=501
filtro_pasbajo = sig.firls(numtaps, freq_pb, gain_pb, fs=fs)

# Mostrar respuesta en frecuencia del filtro combinado
filtro_pbanda = np.convolve(filtro_pasalto, filtro_pasbajo)

w, h = sig.freqz(filtro_pbanda, worN=8000, fs=fs)

plt.figure(figsize=(10,4))
plt.plot(w, 20*np.log10(np.abs(h)+1e-15))
plt.title('Filtro pasa banda por cuadrados minimos (concatenacion)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.ylim([-100, 10])
plt.legend()
plt.grid(True)
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()

# Mejor comportamiento: filtro pasa banda original (ventanas) filtro pasabanda restrinjido. reemplazamos en el diciconario
filtros_fir['Método de cuadrados mínimos'] = filtro_pbanda
filtros_fir['Método de ventanas'] = Filtro_Ventana

#%% Filtrado IIR

ecgs_filtrados = {}  
nombres_filtros = ['Butter', 'Cheby I']

for nombre, filtro in filtros_i.items():
    ecg_filt = sig.sosfilt(filtro, ecg_one_lead)
    ecgs_filtrados[nombre] = ecg_filt
    plt.figure() 
    plt.plot(ecg_one_lead, label= 'Señal sin filtrar')
    plt.plot(ecg_filt, label='Señal filtrada')
    plt.title(f'ECG filtrado - {nombre}')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    plt.legend()
    plt.show()

regs_1 = (
    np.array([5, 5.2]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

demora = 0

for nombre, ecg_filt in ecgs_filtrados.items():
    fig, axs = plt.subplots(1, len(regs_1), figsize=(18, 5), sharey=True)
    axs = axs.flatten()
    
    for i, reg in enumerate(regs_1):
        plt.sca(axs[i])
        plot_regions(ecg_one_lead, ecg_filt, [reg], demora, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)

    plt.suptitle(f'{nombre} - Regiones SIN ruido', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

regs_2 = (
    [4000, 5500],
    [10000, 11000],
)
demora2 = 67

for nombre, ecg_filt in ecgs_filtrados.items():
    fig, axs = plt.subplots(1, len(regs_2), figsize=(12, 5), sharey=True)
    axs = axs.flatten()
    
    for i, reg in enumerate(regs_2):
        plt.sca(axs[i])
        plot_regions(ecg_one_lead, ecg_filt, [reg], demora2, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región ruido {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)

    plt.suptitle(f'{nombre} - Regiones CON ruido', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#%% Filtrado FIR

ecgs_filtrados_fir = {}
nombres_filtros = ['Ventanas', 'Cuadrados Minimos']

for nombre, filtro in filtros_fir.items():
    ecg_filt = np.convolve(ecg_one_lead, filtro, mode='same')
    ecgs_filtrados_fir[nombre] = ecg_filt
    plt.figure() 
    plt.plot(ecg_one_lead, label= 'Señal sin filtrar')
    plt.plot(ecg_filt, label='Señal filtrada')
    plt.title(f'ECG filtrado - {nombre}')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud [V]')
    plt.grid(True)
    plt.legend()
    plt.show()

regs_1 = (
    np.array([5, 5.2]) * 60 * fs,
    np.array([12, 12.4]) * 60 * fs,
    np.array([15, 15.2]) * 60 * fs,
)

demora = 0

for nombre, ecg_filt in ecgs_filtrados_fir.items():
    fig, axs = plt.subplots(1, len(regs_1), figsize=(18, 5), sharey=True)
    axs = axs.flatten()
    
    for i, reg in enumerate(regs_1):
        plt.sca(axs[i])
        plot_regions(ecg_one_lead, ecg_filt, [reg], demora, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)

    plt.suptitle(f'{nombre} - Regiones de interes 1', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

regs_2 = (
    [4000, 5500],
    [10000, 11000],
)
demora2 = 0

for nombre, ecg_filt in ecgs_filtrados_fir.items():
    fig, axs = plt.subplots(1, len(regs_2), figsize=(12, 5), sharey=True)
    axs = axs.flatten()
    
    for i, reg in enumerate(regs_2):
        plt.sca(axs[i])
        plot_regions(ecg_one_lead, ecg_filt, [reg], demora2, label=nombre, crear_figura=False)
        axs[i].set_title(f'{nombre} - Región 2 {i+1}')
        axs[i].set_xlabel('Muestras')
        axs[i].set_ylabel('Amplitud')
        axs[i].grid(True)

    plt.suptitle(f'{nombre} - Regiones de interes 2', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

