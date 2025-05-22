# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:54:17 2025

@author: l
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt

# Frecuencia
omega = np.linspace(0, np.pi, 500)

# Funciones transferencia (respuesta en frecuencia)
sistemas = {
    'H_a': 1 + np.exp(-1j*omega) + np.exp(-2j*omega) + np.exp(-3j*omega),
    'H_b': 1 + np.exp(-1j*omega) + np.exp(-2j*omega) + np.exp(-3j*omega) + np.exp(-4j*omega),
    'H_c': 1 - np.exp(-1j*omega),
    'H_d': 1 - np.exp(-2j*omega)
}

# Coeficientes 

numeradores = {
    'H_a': [1, 1, 1, 1],           
    'H_b': [1, 1, 1, 1, 1],             
    'H_c': [1, -1], 
    'H_d': [1, 0, -1]
}

denominadores = {
    'H_a': [1, 0, 0, 0],
    'H_b': [1, 0, 0, 0, 0],
    'H_c': [1, 0],
    'H_d': [1, 0, 0]
}

# Visualización
for i in sistemas:
    H = sistemas[i]
    H_mod = np.abs(H)
    H_phase = np.angle(H)
    
    num = numeradores[i]
    den = denominadores[i]
    zeros, poles, gain = sig.tf2zpk(num, den)
    
    # Crear figura con 3 subgráficos
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # Módulo
    axs[0].plot(omega, H_mod)
    axs[0].set_title(f'Módulo |H(e^{{jω}})| - {i}')
    axs[0].set_xlabel('ω (radianes)')
    axs[0].set_ylabel('Módulo')
    axs[0].grid(True)
    
    # Fase
    axs[1].plot(omega, H_phase)
    axs[1].set_title(f'Fase ∠H(e^{{jω}}) - {i}')
    axs[1].set_xlabel('ω (radianes)')
    axs[1].set_ylabel('Fase (rad)')
    axs[1].set_xlim([0, np.pi])
    axs[1].grid(True)
    
    # Polos y ceros
    axs[2].scatter(np.real(zeros), np.imag(zeros), marker='o', s=80, label='Ceros', color='blue')
    axs[2].scatter(np.real(poles), np.imag(poles), marker='x', s=80, label='Polos', color='red')
    axs[2].add_artist(plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray'))
    axs[2].set_title(f'Plano-Z - {i}')
    axs[2].set_xlabel('Re')
    axs[2].set_ylabel('Im')
    axs[2].set_aspect('equal')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlim([-1.5, 1.5])
    axs[2].set_ylim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.show()
