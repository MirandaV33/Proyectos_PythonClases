# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 17:59:38 2025

@author: l
"""

#%% Librerias
import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

# Funciones para filtros
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline 
from scipy.signal import correlate, find_peaks
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

#%% Diseño de filtro
#%% Mediana

# x_hat =s - b_hat 

#Segmento de interes: 
fs = 1000 # Hz
ecg_segment= ecg_one_lead[700000:745000]
t_segment=np.linspace(0, len(ecg_one_lead)/ fs, len(ecg_segment))

#Ventana
win1=201
win2=1201

# Filtro de mediana (200ms)
base_line= sig.medfilt(ecg_segment, kernel_size=win1)

# Filtro de mediana (600ms)
base_line2=sig.medfilt(base_line, kernel_size=win2)

#Filtramos la señal
x_hat = ecg_segment - base_line2

#Visualizacion
plt.figure(figsize=(12,5)) 
plt.plot(t_segment,ecg_segment, label= 'Señal sin filtrar')
plt.plot(t_segment, base_line2, label='Linea de base por mediana')
plt.plot(t_segment, x_hat, label='Señal Filtrada')
plt.title("ECG: Electrocardiograma")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()


#%% Interpolación 

n0 = int(0.1 * fs) # --> Defnimos en funcion de fs para que este muestreada a la misma frecuencia el baseline
ecg_segment= ecg_one_lead[700000:745000]

# Calculo los mi = ni - n0 (segmento PQ) ni punto maximo detecto el qrs para asi desfasarme y encontrar el nivel isoelectrico
# --> Estos son los tiempos (en muestras) a las que ocurren las detecciones del qrs
m_i = qrs_detections - n0
m_i = m_i[(m_i >= 0) & (m_i < len(ecg_segment))]

# ---> Esta es la señal a esos tiempos m_i
s_mi = ecg_segment[m_i]

# Interpolamos con spline cúbico
spline = CubicSpline(m_i, s_mi)
# Evaluamos el spline en todos los puntos 
n = np.arange(len(ecg_segment))
base_spline = spline(n)

# Filtramos la señal 
x_hat2 = ecg_segment - base_spline

plt.figure(figsize=(12, 5))
plt.plot(ecg_segment, label='Señal original')
plt.plot(base_spline, label='Línea de base por Splines')
plt.plot(x_hat2, label='Señal filtrada')
plt.legend()
plt.title("ECG y línea de base estimada")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()

#%% Match Filter
# Segmento de ECG
ecg_segment = ecg_one_lead[300000:312000]

# Invierto el patrón del filtro adaptado
h = qrs_pattern1[::-1]

# Correlación en modo 'same' para que tenga el mismo largo que ecg_segment
correlation = correlate(ecg_segment, h, mode='same')

# Detección de picos en la correlación
peaks, _ = find_peaks(correlation, height=np.max(correlation)*0.3, distance=int(fs*0.6))

# Picos verdaderos dentro del segmento (convertidos a relativo al segmento)
peaks_true = qrs_detections[(qrs_detections >= 300000) & (qrs_detections < 312000)] - 300000


# Visualizacion 
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

axs[0].plot(ecg_segment, label='ECG')
axs[0].plot(peaks_true, ecg_segment[peaks_true], 'go', label='Latidos reales')
axs[0].plot(peaks, ecg_segment[peaks], 'rx', label='Latidos detectados')
axs[0].set_title('Señal ECG (latidos reales)')
axs[0].set_xlabel('Muestras')
axs[0].set_ylabel('Amplitud')
axs[0].grid(True)
axs[0].legend()

# Correlación con detección de picos
axs[1].plot(correlation, label='Correlación (matched filter)')
axs[1].plot(peaks_true, correlation[peaks_true], 'go', label='Latidos reales')
axs[1].plot(peaks, correlation[peaks], 'ro', label='Latidos detectados')
axs[1].set_title('Filtro Adaptado - Correlación y Detecciones')
axs[1].set_xlabel('Muestras')
axs[1].set_ylabel('Amplitud')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()


# Métricas
tolerancia = int(0.15 * fs)  # 150 ms de tolerancia

TP = 0
FP = 0
FN = 0

# Creamos array para marcar qué latidos reales ya fueron emparejados
true_matched = np.zeros_like(peaks_true, dtype=bool)

# Comparar cada pico detectado con los verdaderos
for p in peaks:
    match = False
    for i, pt in enumerate(peaks_true):
        if not true_matched[i] and abs(p - pt) <= tolerancia:
            TP += 1
            true_matched[i] = True
            match = True
            break
    if not match:
        FP += 1

# Lo que no fue emparejado se considera FN
FN = np.sum(~true_matched)

# Cálculo de métricas
if TP + FN > 0:
    sensibilidad = TP / (TP + FN)
else:
    sensibilidad = 0.0

if TP + FP > 0:
    precision = TP / (TP + FP)
else:
    precision = 0.0

# Mostrar resultados
print(f"Latidos reales detectados correctamente (TP): {TP}")
print(f"Falsos positivos (FP): {FP}")
print(f"Falsos negativos (FN): {FN}")
print(f"Sensibilidad (Recall): {sensibilidad:.2%}")
print(f"Precisión (PPV): {precision:.2%}")

