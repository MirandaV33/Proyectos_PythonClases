# -*- coding: utf-8 -*-
"""
Created on Sun May 11 11:45:52 2025

@author: l
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

#%% Definicion de funciones 

def vertical_flaten(a):

    return a.reshape(a.shape[0],1)

def verificar_parseval(signal, nombre):
    area = np.sum(signal**2)
    #IMPORTANTE:Como mi señal se extrae con dimensiones [***, 1] mi fft no va a reconocer estas dimensiones porque trabaja por columnas, entonces va a interpretar que hay una sola columna! 
    #Conertimos el vector [N,1] a un vector de una sola dimension.
    ft = np.fft.fft(signal.flatten())
    parseval = np.mean(np.abs(ft)**2)
    error = np.abs(area - parseval) / area

    se_cumple = error < 0.01 
    
    return area, parseval, error, se_cumple

def calcular_anchodebanda(f, Pxx):
    # Paso 1: Potencia total acumulada
    potencia_total = np.sum(Pxx)
    
    # Paso 2: Potencia acumulada
    potencia_acumulada = np.cumsum(Pxx)
    
    # Paso 3: Umbrales 95% y 98%
    umbral_95 = 0.95 * potencia_total
    umbral_98 = 0.98 * potencia_total
    
    # Paso 4: Buscar índices donde se alcanza el umbral
    idx_95 = np.argmax(potencia_acumulada >= umbral_95)
    idx_98 = np.argmax(potencia_acumulada >= umbral_98)
    
    # Paso 5: Ancho de banda
    BW_95 = f[idx_95]
    BW_98 = f[idx_98]
    
    return BW_95, BW_98


#%% Extraccion de datos de las señales y procesamiento

#%% ECG 

fs_ecg = 1000 # Hz

#### ECG CON RUIDO ###

#Visualizamos lo que contiene el archivo
#io.whosmat('ECG_TP4.mat')

#Guardamos los datos del archivo en un diccionario de Python: enlistamos
mat_struct = sio.loadmat('./ECG_TP4.mat')

#Extraemos los datos de la señal y los metemos en un vector.
ecg_one_lead = vertical_flaten(mat_struct['ecg_lead'])
N1 = len(ecg_one_lead)

# Normalizo la señal 
ecg_one_lead_r = ecg_one_lead / np.std(ecg_one_lead)

### ECG SIN RUIDO ###
ecg_one_lead = np.load('ecg_sin_ruido.npy')
N2 = len(ecg_one_lead)
ecg_one_lead= ecg_one_lead / np.std(ecg_one_lead)


# %% Estimacion por Welch

f_ecg_r, Pxx_ecg_r = sig.welch(ecg_one_lead_r, fs_ecg, nfft=N1, window='hann', nperseg=N1//6, axis=0)
f_ecg, Pxx_ecg = sig.welch(ecg_one_lead, fs_ecg, nfft=N2, window='hann', nperseg=N2//6, axis=0)
print(np.max(Pxx_ecg_r))
print(np.max(Pxx_ecg))

# %%Visualizacion de resultados 

###SEÑAL TEMPORAL###

plt.figure(figsize=(10, 6))
#ECG con ruido
plt.subplot(2, 1, 1)  
plt.plot(ecg_one_lead_r[5000:12000])
plt.title("ECG: Electrocardiograma con ruido")

# ECG sin ruido
plt.subplot(2, 1, 2)
plt.plot(ecg_one_lead[5000:12000])
plt.title("ECG: Electrocardiograma sin ruido")

plt.tight_layout()
plt.show()

### ESPECTRO ###
#Grilla de frecuencias  
bfrec1= f_ecg_r <= fs_ecg / 2
bfrec2= f_ecg <= fs_ecg / 2

plt.figure(figsize=(10, 6))
#ECG con ruido
plt.subplot(2, 1, 1)  
plt.plot(f_ecg_r, 10 * np.log10(2 * np.abs(Pxx_ecg_r[bfrec1])**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of: "ECG con ruido')

plt.subplot(2, 1, 2)  
plt.plot(f_ecg, 10 * np.log10(2 * np.abs(Pxx_ecg[bfrec2])**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of: "ECG sin ruido')

plt.tight_layout()
plt.show()

# %% Verifico Parseval
area_r, parseval_r, error_r, cumple_r= verificar_parseval(ecg_one_lead_r, "ECG con ruido")
area, parseval, error, cumple = verificar_parseval(ecg_one_lead, "ECG sin ruido")

# %% Busco Ancho de Banda
# Asumimos comportamiento pasa bajos, asi tomamos el area acumulando la potencia con Cumsum. Es decir, toma el 95% del area bajo la curva desde 0 Hz
BW_95_ecg_r, BW_98_ecg_r = calcular_anchodebanda(f_ecg_r, Pxx_ecg_r)
BW_95_ecg, BW_98_ecg = calcular_anchodebanda(f_ecg, Pxx_ecg)

# %% Visualizacion de resultados

#Tabla con los resultados ordenados
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
tabla = [
    ["Señal", "Energía Tiempo", "Energía Frecuencia", "Error Relativo", "¿Se cumple Parseval?", "BW 95%", "BW 98%"],
    ["ECG con ruido", f"{area_r:.2f}", f"{parseval_r:.2f}", f"{error_r*100:.2f}%", "Sí" if cumple_r else "No", f"{BW_95_ecg_r:.1f}", f"{BW_98_ecg_r:.1f}" ],
    ["ECG sin ruido", f"{area:.2f}", f"{parseval:.2f}", f"{error*100:.2f}%", "Sí" if cumple else "No", f"{BW_95_ecg:.1f}", f"{BW_98_ecg:.1f}" ],
    ]
tabla_plot = ax.table(cellText=tabla, loc='center', cellLoc='center', colWidths=[0.14]*7)
tabla_plot.auto_set_font_size(False)
tabla_plot.set_fontsize(10)
tabla_plot.scale(1, 2)

plt.title("Verificación de Parseval y calculo de Ancho de Banda (por Welch)")
plt.show()

# Grafico de la densidad espectral de potencia con el BW marcado. ç

plt.figure(figsize=(10, 6))
#ECG con ruido
plt.subplot(2, 1, 1)  
plt.plot(f_ecg_r, 10 * np.log10(2 * np.abs(Pxx_ecg_r[bfrec1])**2))
plt.axvline(BW_95_ecg_r, color='r', linestyle='--', label=f'BW 95%: {BW_95_ecg_r:.1f} Hz')
plt.axvline(BW_98_ecg_r, color='orange', linestyle='--', label=f'BW 98%: {BW_98_ecg_r:.1f} Hz')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of "ECG" ')
plt.legend()
# ECG sin ruido 
plt.subplot(2, 1, 2)
plt.plot(f_ecg, 10 * np.log10(2 * np.abs(Pxx_ecg[bfrec2])**2))
plt.axvline(BW_95_ecg, color='r', linestyle='--', label=f'BW 95%: {BW_95_ecg:.1f} Hz')
plt.axvline(BW_98_ecg, color='orange', linestyle='--', label=f'BW 98%: {BW_98_ecg:.1f} Hz')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of "ECG" ')
plt.legend()

plt.tight_layout()
plt.show()

#%% PPG
 
fs_ppg=1000 #Hz

# Extraigo los datos del archivo
ppg = np.load('ppg_sin_ruido.npy')
Np=len(ppg)

#Normalizo
ppg= ppg/np.std(ppg)

# %% Estimacion de resultados
f_ppg, Pxx_ppg = sig.welch(ppg, fs_ppg, nfft=Np, window='hann', nperseg=Np//6, axis=0)

# %% Visualizacion de resultados

###Señal Temporal###
plt.figure()  
plt.plot(ppg[0:1000])
plt.title("PPG: Pletismografía")

### Estrectro ###
bfrecp= f_ppg <= fs_ppg / 2

plt.figure()
plt.plot(f_ppg, 10 * np.log10(2 * np.abs(Pxx_ppg[bfrecp])**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of "PPG" ')
plt.legend()

# %% Verifico Parseval y calculo Ancho de banda

area_p, parseval_p, error_p, cumple_p = verificar_parseval(ppg, "PPG")
BW_95_ppg, BW_98_ppg = calcular_anchodebanda(f_ppg, Pxx_ppg)

# %% Visualizacion de resultados 

# Resultados obtenidos 
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
tabla = [
    ["Señal", "Energía Tiempo", "Energía Frecuencia", "Error Relativo", "¿Se cumple Parseval?", "BW 95%", "BW 98%"],
    ["PPG sin ruido", f"{area_p:.2f}", f"{parseval_p:.2f}", f"{error_p*100:.2f}%", "Sí" if cumple_p else "No", f"{BW_95_ppg:.1f}", f"{BW_98_ppg:.1f}" ],
    ]
tabla_plot = ax.table(cellText=tabla, loc='center', cellLoc='center', colWidths=[0.14]*7)
tabla_plot.auto_set_font_size(False)
tabla_plot.set_fontsize(10)
tabla_plot.scale(1, 2)

plt.title("Verificación de Parseval y calculo de Ancho de Banda (por Welch)")
plt.show()


# Grafico espectral marcando los ancho de banda
plt.figure()
plt.plot(f_ppg, 10 * np.log10(2 * np.abs(Pxx_ppg)**2))
plt.axvline(BW_95_ppg, color='r', linestyle='--', label=f'BW 95%: {BW_95_ppg:.1f} Hz')
plt.axvline(BW_98_ppg, color='orange', linestyle='--', label=f'BW 98%: {BW_98_ppg:.1f} Hz')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of "PPG"')
plt.legend()
plt.show()


#%% AUDIO 

fs_audio_c, wav_data_c = sio.wavfile.read('la cucaracha.wav')
fs_audio_s, wav_data_s = sio.wavfile.read('silbido.wav')
fs_audio_p, wav_data_p = sio.wavfile.read('prueba psd.wav')

N_c = len(wav_data_c)
N_s = len(wav_data_s)
N_p = len(wav_data_p)

# Normalizo en potencia: máximo entorno a -40dB
wavvv_data_c = wav_data_c / np.std(wav_data_c)
wavvv_data_s = wav_data_s / np.std(wav_data_s)
wavvv_data_p = wav_data_p / np.std(wav_data_p)

# Normalizo 2: Respecto al máximo, esto es para que todos tengan su pico al 0dB
# El ancho de banda nos juega en contra para comparar espectros de diferentes frecuencias
wavvv_data_c = wav_data_c / np.max(wav_data_c)
wavvv_data_s = wav_data_s / np.max(wav_data_s)
wavvv_data_p = wav_data_p / np.max(wav_data_p)

# %% Estimación por Welch

f_c, Pxx_c = sig.welch(wavvv_data_c, fs_audio_c, nfft=N_c, window='hann', nperseg=N_c//6, axis=0)
f_s, Pxx_s = sig.welch(wavvv_data_s, fs_audio_s, nfft=N_s, window='hann', nperseg=N_s//6, axis=0)
f_p, Pxx_p = sig.welch(wavvv_data_p, fs_audio_p, nfft=N_p, window='hann', nperseg=N_p//6, axis=0)

# %% Visualización de resultados

#### Señal Temporal ####
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(wavvv_data_c)
plt.title("Audio.wav: La cucaracha")

plt.subplot(3, 1, 2)
plt.plot(wavvv_data_s)
plt.title("Audio.wav: Silbido")

plt.subplot(3, 1, 3)
plt.plot(wavvv_data_p)
plt.title("Audio.wav: Prueba")

plt.tight_layout()
plt.show()


##### ESPECTRO #######

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(f_c, 10 * np.log10(2 * np.abs(Pxx_c)**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of audio.wav of: "La cucaracha"')

plt.subplot(3, 1, 2)
plt.plot(f_s, 10 * np.log10(2 * np.abs(Pxx_s)**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of audio.wav: "Silbido"')

plt.subplot(3, 1, 3)
plt.plot(f_p, 10 * np.log10(2 * np.abs(Pxx_p)**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of audio.wav: "Prueba"')

plt.tight_layout()
plt.show()

# %% Verifico Parseval

area_c, parseval_c, error_c, cumple_c = verificar_parseval(wavvv_data_c, "La Cucaracha")
area_s, parseval_s, error_s, cumple_s = verificar_parseval(wavvv_data_s, "Silbido")
area_p, parseval_p, error_p, cumple_p = verificar_parseval(wavvv_data_p, "Prueba")

# Muestro Resultados
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
tabla = [
    ["Señal", "Energía Tiempo", "Energía Frecuencia", "Error Relativo", "¿Se cumple Parseval?"],
    ["La cucaracha", f"{area_c:.2f}", f"{parseval_c:.2f}", f"{error_c*100:.2f}%", "Sí" if cumple_c else "No"],
    ["Silbido", f"{area_s:.2f}", f"{parseval_s:.2f}", f"{error_s*100:.2f}%", "Sí" if cumple_s else "No"],
    ["Prueba", f"{area_p:.2f}", f"{parseval_p:.2f}", f"{error_p*100:.2f}%", "Sí" if cumple_p else "No"]
]

tabla_plot = ax.table(cellText=tabla, loc='center', cellLoc='center', colWidths=[0.25]*5)
tabla_plot.auto_set_font_size(False)
tabla_plot.set_fontsize(10)
tabla_plot.scale(1, 2)

plt.title("Verificación de Parseval (por Welch)")
plt.show()


# %% Calculamos el ancho de banda 

# Asumimos comportamiento pasa bajos, asi tomamos el area acumulando la potencia con Cumsum. Es decir, toma el 95% del area bajo la curva desde 0 Hz
BW_95_c, BW_98_c = calcular_anchodebanda(f_c, Pxx_c)
BW_95_s, BW_98_s = calcular_anchodebanda(f_s, Pxx_s)
BW_95_p, BW_98_p = calcular_anchodebanda(f_p, Pxx_p)

# Mostrar resultados

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')
tabla = [
    ["Señal", "BW 95% [Hz]", "BW 98% [Hz]"],
    ["La cucaracha", f"{BW_95_c:.1f}", f"{BW_98_c:.1f}"],
    ["Silbido", f"{BW_95_s:.1f}", f"{BW_98_s:.1f}"],
    ["Prueba", f"{BW_95_p:.1f}", f"{BW_98_p:.1f}"]
]
tabla_plot = ax.table(cellText=tabla, loc='center', cellLoc='center', colWidths=[0.3, 0.3, 0.3])
tabla_plot.auto_set_font_size(False)
tabla_plot.set_fontsize(12)
tabla_plot.scale(1, 2)
plt.title("Ancho de banda de las señales (por Welch)")
plt.show()

# %% Visualizacion de resultados
# Grafico de la densidad espectral de potencia con el BW marcado. 

plt.figure(figsize=(10, 6))
plt.subplot(3,1,1)
plt.plot(f_c, 10 * np.log10(2 * np.abs(Pxx_c)**2))
plt.axvline(BW_95_c, color='r', linestyle='--', label=f'BW 95%: {BW_95_c:.1f} Hz')
plt.axvline(BW_98_c, color='orange', linestyle='--', label=f'BW 98%: {BW_98_c:.1f} Hz')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of audio.wav of: "La cucaracha"')
plt.legend()

plt.subplot(3,1,2)
plt.plot(f_s, 10 * np.log10(2 * np.abs(Pxx_s)**2))
plt.axvline(BW_95_s, color='r', linestyle='--', label=f'BW 95%: {BW_95_s:.1f} Hz')
plt.axvline(BW_98_s, color='orange', linestyle='--', label=f'BW 98%: {BW_98_s:.1f} Hz')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of audio.wav of: "Silbido"')
plt.legend()

plt.subplot(3,1,3)
plt.plot(f_p, 10 * np.log10(2 * np.abs(Pxx_p)**2))
plt.axvline(BW_95_p, color='r', linestyle='--', label=f'BW 95%: {BW_95_p:.1f} Hz')
plt.axvline(BW_98_p, color='orange', linestyle='--', label=f'BW 98%: {BW_98_p:.1f} Hz')
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density of audio.wav of: "Prueba"')
plt.legend()

plt.tight_layout()
plt.show()
