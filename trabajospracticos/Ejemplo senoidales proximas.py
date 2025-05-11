# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 19:22:04 2025

@author: l
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
f1 = 1000      # Frecuencia 1 en Hz
f2 = 1020      # Frecuencia 2 en Hz
fs = 5000      # Frecuencia de muestreo en Hz
T = 0.5        # Tiempo total de observación en segundos
t = np.linspace(0, T, int(fs*T), endpoint=False)  # Vector de tiempos

# Señal: suma de dos senoidales
signal = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

# FFT
N = len(signal)
freqs = np.fft.fftfreq(N, 1/fs)
fft_signal = np.fft.fft(signal)
fft_magnitude = np.abs(fft_signal)/N

# Solo nos interesa la mitad positiva
mask = freqs > 0

# Gráficos
plt.figure(figsize=(12,6))

# Señal en el tiempo
plt.subplot(2,1,1)
plt.plot(t, signal)
plt.title('Señal en el tiempo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()

# Espectro
plt.subplot(2,1,2)
plt.plot(freqs[mask], 2*fft_magnitude[mask])  # Multiplico por 2 para conservar energía
plt.title('Espectro de magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.grid()
plt.xlim(900, 1100)  # Zoom alrededor de las frecuencias de interés

plt.tight_layout()
plt.show()
