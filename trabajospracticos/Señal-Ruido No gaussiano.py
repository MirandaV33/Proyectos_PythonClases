# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 14:17:53 2025

@author: l
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import square

fs = 1000  # frecuencia de muestreo
t = np.linspace(0, 1, fs, endpoint=False)

# Señal: 50 Hz senoidal
signal = np.sin(2 * np.pi * 50 * t)

non_gaussian_noise = 0.5 * square(2 * np.pi * 50 * t)
gaussian_noise = 0.5 * np.random.normal(0, 1, size=t.shape)

# FFT
N = len(t)
frequencies = fftfreq(N, 1/fs)

fft_signal = np.abs(fft(signal))
fft_noise = np.abs(fft(non_gaussian_noise))
fft_gauss = np.abs(fft(gaussian_noise))

# Solo frecuencias positivas
bfrec = frequencies >= 0

# Gráfico
plt.figure(figsize=(12, 5))
plt.plot(frequencies[bfrec], 20 * np.log10(2 * (fft_signal[bfrec])**2), label='Señal', alpha=0.7)
plt.plot(frequencies[bfrec], 20 * np.log10(2 * (fft_noise[bfrec])**2), label='Ruido no gaussiano', alpha=0.7)
plt.plot(frequencies[bfrec], 20 * np.log10(2 * fft_gauss[bfrec]**2), label='Ruido gaussiano (blanco)', alpha=0.7)
plt.xlabel('Frecuencia [Hz]')
plt.xlim(30,70)
plt.ylim(-60, 150)
plt.ylabel('Magnitud [dB]')
plt.title('Espectro de señal y ruido')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
