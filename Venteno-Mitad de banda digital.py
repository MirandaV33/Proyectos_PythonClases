# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:27:08 2025

@author: l
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

#%% módulos y funciones a importar

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    
    ts = 1/fs  # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn)  # Cambié N por nn
    
    # Grilla de amplitud
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc  # Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx

#%% Datos de la simulación

fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
#Normalizamos la resolucion espectral. 

ts = 1/fs # tiempo de muestreo
df =  fs/N # resolución espectral

# %% Llamada a la función y procesamiento

tt, xx = mi_funcion_sen(vmax=2, dc=0, ff=N/4 +0.5, ph=0, nn=N, fs=fs)

#Normalizo 
xn= xx/np.std(xx)

# Defino la ventana 
M=N
w= signal.windows.barthann(M)

# Convoluciono 
xw= xn * w

#Normalizo 
xwn = xw / np.std(xw)

#%% Visualizacon de resultados

#Representacion temporal
plt.figure(1)
plt.plot(tt, xx)
plt.title('Representacion temporal' )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

#Representacion espectral 

plt.figure(2)
ft_xn = 1/N * np.fft.fft(xn)  
ft_xwn = 1/N * np.fft.fft(xwn) 

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)
bfrec = ff <= fs/2 #Vector de n valores que hace que me quede con LA MITAD el vector, nos devuelve un verdadero o falso segun la condicion 

plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xn[bfrec])**2), label="Señal original", color='orange')
plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xwn[bfrec])**2), label="Señal con ventana")
plt.title('Representacion espectral' )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()