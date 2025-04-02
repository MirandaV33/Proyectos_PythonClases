# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 16:36:21 2025

@author: l
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros generales
f_s = 10000  # Frecuencia de muestreo en Hz
frecuencias = np.linspace(1, f_s / 2, 1000)  # Rango de frecuencias de 1 Hz a Nyquist

# Parámetros de los filtros
f_c = 1000  # Frecuencia de corte para pasa bajos y pasa altos
f_c1, f_c2 = 500, 1500  # Frecuencias de corte para pasa banda y notch

# Funciones de transferencia
# Pasa Bajos
H_pb = (2j * np.pi * f_c) / (2j * np.pi * frecuencias + 2j * np.pi * f_c)

# Pasa Altos
H_pa = (2j * np.pi * frecuencias) / (2j * np.pi * frecuencias + 2j * np.pi * f_c)

# Pasa Banda
H_pbanda = ((2j * np.pi * frecuencias) / (2j * np.pi * frecuencias + 2j * np.pi * f_c1)) * \
           ((2j * np.pi * f_c2) / (2j * np.pi * frecuencias + 2j * np.pi * f_c2))

# Notch (Rechaza Banda)
H_notch = 1 - H_pbanda

# Convertimos a dB
H_pb_dB = 20 * np.log10(np.abs(H_pb))
H_pa_dB = 20 * np.log10(np.abs(H_pa))
H_pbanda_dB = 20 * np.log10(np.abs(H_pbanda))
H_notch_dB = 20 * np.log10(np.abs(H_notch))

# Gráfico Pasa Bajos
plt.figure(figsize=(8, 6))
plt.plot(frecuencias, H_pb_dB, label='Filtro Pasa Bajos', color='blue')
plt.xscale('log')
plt.ylim([-60, 5])
plt.title('Filtro Pasa Bajos (dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()

# Gráfico Pasa Altos
plt.figure(figsize=(8, 6))
plt.plot(frecuencias, H_pa_dB, label='Filtro Pasa Altos', color='red')
plt.xscale('log')
plt.ylim([-60, 5])
plt.title('Filtro Pasa Altos (dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()

# Gráfico Pasa Banda
plt.figure(figsize=(8, 6))
plt.plot(frecuencias, H_pbanda_dB, label='Filtro Pasa Banda', color='green')
plt.xscale('log')
plt.ylim([-60, 5])
plt.title('Filtro Pasa Banda (dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()

# Gráfico Notch (Rechaza Banda)
plt.figure(figsize=(8, 6))
plt.plot(frecuencias, H_notch_dB, label='Filtro Notch (Rechaza Banda)', color='purple')
plt.xscale('log')
plt.ylim([-60, 5])
plt.title('Filtro Notch (Rechaza Banda) (dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Ganancia (dB)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()

