# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:02:20 2025

@author: l
"""

#%%  Inicialización de librerias 

import numpy as np 
import matplotlib.pyplot as plt 

#%%  Generación de señales

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    
    ts = 1/fs # tiempo de muestreo
    
    # grilla de sampleo temporal
    tt = np.linspace(0, (N-1)*ts, N) #Grilla total de tiempo 
    
    ##Grilla de amplitud
    xx = vmax*np.sin(2*np.pi*ff*tt+ph) + dc #Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx

def mi_funcion_triangular(vmax, dc, ff, ph, nn, fs):
    ts = 1 / fs  # Tiempo de muestreo
    tt = np.linspace(0, (nn - 1) * ts, nn)  # Grilla de tiempo
    xx = vmax * (2 * np.abs((2 * (tt * ff + ph / (2 * np.pi)) % 1) - 1) - 1) + dc
    return tt, xx

##Parámetros 
fs= 1000#Frecuencia de muestreo 
N= 1000 #Cantidad de muestras

#%% Generacion de señales de prueba

##Para una mejor visualizacion, le agrego una señal continua para desplazar las funciones en el grafico y se vean todas. 
tt1, xx1 = mi_funcion_sen( vmax = 1, dc = 2, ff = 1, ph=0, nn = N, fs = fs)
tt2, xx10= mi_funcion_sen( vmax = 1, dc = 0, ff = 10, ph=0, nn = N, fs = fs)
tt3, xx500 = mi_funcion_sen( vmax = 1, dc = 2, ff = 500, ph=0, nn = N, fs = fs)
tt4, xx999 = mi_funcion_sen( vmax = 1, dc = 4, ff = 999, ph=0, nn = N, fs = fs)
tt5, xx1001 = mi_funcion_sen( vmax = 1, dc = 4, ff = 1001, ph=0, nn = N, fs = fs)
tt6, xx2001 = mi_funcion_sen( vmax = 1, dc = 0, ff = 2001, ph=0, nn = N, fs = fs)
tt7, xxdesfasada = mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=np.pi/2, nn = N, fs = fs)

tt, xx= mi_funcion_sen( vmax = 1, dc = 0, ff = 1, ph=0, nn = N, fs = fs)
tt8, xx8=  mi_funcion_triangular(vmax = 1, dc = 0, ff = 1, ph=0 , nn = N, fs = fs)

#%%  Presentación gráfica de resultados

plt.figure(1)
plt.plot(tt1, xx1, label='f0 = 1 Hz, ph = 0', color='b')
plt.plot(tt2, xx10, label='f0 = 10 Hz, ph = 0', color='y')
plt.plot(tt3, xx500, label='f0 = 500 Hz, ph = 0', color='g')
plt.plot(tt4, xx999, label='f0 = 999 Hz, ph = 0', color='r')
plt.plot(tt5, xx1001, label='f0 = 1001 Hz, ph = 0', color='c')
plt.plot(tt6, xx2001, label='f0 = 2001 Hz, ph = 0', color='m')
plt.plot(tt7, xxdesfasada, label='f0 = 1 Hz, ph = π/2', color='k')
plt.title('Señal: ')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(tt, xx, label='Señal senoidal', color='b')
plt.plot(tt8, xx8, label='Señal triangular', color='m')
plt.title('Señal: ')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud')
plt.legend()
plt.show()

