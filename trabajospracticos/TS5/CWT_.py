# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:34:06 2025

@author: l
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import fftconvolve, convolve
from scipy.signal import stft, welch
import pywt


#%% audio
fs=1000
fs_audio_c, wav_data_c = sio.wavfile.read('la cucaracha.wav')
N_c = len(wav_data_c)
t = np.arange(N_c) / fs

scales = np.logspace(0, np.log10(150), num=100)  # 1 a 100 en logscale, pero igual serán convertidas a Hz

wavelet = pywt.ContinuousWavelet('cmor1.5-1.0') # Podemos observar que esta ondita se parece mas a la ondita de la cucharacha  por eso se ve con mucha mas resolucion 
# wavelet = pywt.ContinuousWavelet('mexh')
#wavelet = pywt.ContinuousWavelet('gaus3')

f_c = pywt.central_frequency(wavelet)  # devuelve frecuencia normalizada
Δt = 1.0 / fs
frequencies = f_c / (scales * Δt)

coefficients, frec = pywt.cwt(wav_data_c, scales, wavelet, sampling_period=Δt)

# Crear figura y ejes
fig = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# Señal
ax1.plot(wav_data_c)
ax1.set_title("Audio: 'La cucaracha'")
ax1.set_ylabel("Amplitud")
plt
pcm = ax2.imshow(np.abs(coefficients),
           extent=[t[0], t[-1], scales[-1], scales[0]],  # nota el orden invertido para eje Y
           cmap='viridis', aspect='auto')
ax2.set_title("CWT con wavelet basada en $B_3(x)$")
ax2.set_xlabel("Tiempo")
ax2.set_ylabel("Escala")
cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # [left, bottom, width, height]
fig.colorbar(pcm, cax=cbar_ax, label="Magnitud")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Dejar espacio para colorbar a la derecha
plt.show()

