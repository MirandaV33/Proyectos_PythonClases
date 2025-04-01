# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:15:57 2025

@author: l
"""
import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen (vmax, dc, ff, ph, nn, Ts):
    
    Ts = 1/Fs  # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*Ts, nn)  # Cambié N por nn
    
    # Grilla de amplitud
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc  # Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx

# Parámetros
ff = 5  # Frecuencia de la señal (Hz)
Fs = 50  # Frecuencia de muestreo (Hz)

T0 = 1 / ff # Período de la señal
N = int(Fs * 2 * T0)  # Número de muestras en dos períodos
t = np.linspace(0, 2 * T0, N, endpoint=False)  # Tiempo discreto en dos períodos
signal = np.cos(2 * np.pi * ff * t)  # Señal cosenoidal

# Tiempo continuo para la señal original
t_fine = np.linspace(0, 2 * T0, 1000)
signal_fine = np.cos(2 * np.pi * ff * t_fine)

# Graficar la señal continua
plt.plot(t_fine, signal_fine, 'b-', label='Señal Continua')

# Graficar las muestras discretas
plt.plot(t, signal, 'ro', markersize=6, label='Muestras Discretas')

# Dibujar líneas desde cada punto de la señal muestreada hacia la señal continua
for i in range(len(t)):
    plt.plot([t[i], t[i]], [0, signal[i]], 'r-', linewidth=1)

# Marcar N y sus réplicas
n_index = len(t) // 2  # Punto central como referencia
plt.axvline(t[n_index], color='g', linestyle='--', linewidth=1, label='N')
if n_index > 0:
    plt.axvline(t[n_index - 1], color='purple', linestyle='-.', linewidth=1, label='N-1')
if n_index < len(t) - 1:
    plt.axvline(t[n_index + 1], color='purple', linestyle='-.', linewidth=1, label='N+1')

# Marcar Fs
plt.axvline(Ts, color='black', linestyle=':', linewidth=1, label='Fs')

# Etiquetas y título
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Señal Muestreada y Continua en Dos Períodos")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()
