# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 21:02:54 2025

@author: l
"""

import numpy as np 
import matplotlib.pyplot as plt 

#%% módulos y funciones a importar

# Senoidal cuya potencia tenga 1Watt
# Encontrar la amplitud de la senoidal que la varianza sea unitaria o --> Normalizar

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    
    ts = 1/fs  # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (nn-1)*ts, nn)  # Cambié N por nn
    
    # Grilla de amplitud
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc  # Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx

# %% Llamada a la función y procesamiento
# tt, xx = mi_funcion_sen(2, 0, 1, 0, 1000, 1000)
tt, xx2 = mi_funcion_sen(2, 0, 1.5, 0, 1000, 1000)
tt, xx3 = mi_funcion_sen(2, 0, 501.5, 0, 1000, 1000)
# tt, xx4 = mi_funcion_sen(2, 0, 499, 0, 1000, 1000)

#%% Datos de la simulación

fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
#Normalizamos la resolucion espectral. 

ts = 1/fs # tiempo de muestreo
df =  fs/N # resolución espectral

#%% Visualizacion

#Representacion temporal 

# plt.plot(tt, xx, label = 'S=1hz')
plt.plot(tt, xx2, label = 'S=1.5')
plt.plot(tt, xx3, label = 'S=500,5')
# plt.plot(tt, xx4, label = 'S=499')

plt.title('Representacion temporal' )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()

#Representacion espectral 

plt.figure(2)
# ft_xx = 1/N*np.fft.fft(xx) 
ft_xx2 = 1/N*np.fft.fft(xx2)
ft_xx3 = 1/N*np.fft.fft(xx3)
# ft_xx4 = 1/N*np.fft.fft(xx4)

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 #Vector de n valores que hace que me quede con LA MITAD el vector, nos devuelve un verdadero o falso segun la condicion 

# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx[bfrec])**2), color='orange')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx2[bfrec])**2), color='black')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx3[bfrec])**2))
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_xx4[bfrec])**2))

plt.title('Representacion espectral' )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()