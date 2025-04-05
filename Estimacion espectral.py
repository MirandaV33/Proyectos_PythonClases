# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 20:07:37 2025

@author: l

"""


import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as pltw

#%% Datos de la simulacion, definicion de constantes

Np=1000 
SNR= 10  #dB
R=200 #numero de muestras
a1=1/np.sqrt(2) #Amplitud de la señal, eligiendo esta ya estoy normalizando la señal. Ya no hace falta dividirla por la desviacion estandar
fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
ts = 1/fs  # tiempo de muestreo
df= fs/N #resolucion espectral 
omega0= fs/4

#%% Generacion de la señal X
# Defino S

# Grilla de sampleo temporal ---> DISRETIZACION DEL TIEMPO (muestreo)
# Vector flat, [0x100]. Tengo que cambiarle las dimensiones con el reshape de 1000x200. Apilado horizontal: numpy.hstack 
tt = np.linspace(0, (N-1)*ts, N).reshape((1000, 1)) 
tt= np.tile(tt, (1, R)) #Repetidor

# Grilla sampleo frecuencial
ff= np.linspace(0, (N-1)*df, N).reshape(1, 1000)

fr = np.random.uniform(-1/2, 1/2, size=(1,R))

omega1= omega0 + fr* (df)

#Argumento de la matriz S-> Se apila verticalmente
#Tengo que reemplazar ff por una matriz de 200 realizaciones [1,200]

S= a1*np.sin(omega1*tt)

# #%% Datos del ruido
# # Señal analogica --> Lo saco de SNR
pot_ruido_analog = 10**(- SNR / 10)
sigma= np.sqrt(pot_ruido_analog)
# #Generacion de ruido analogico 
nn = np.random.normal(0, sigma, (Np, R))
 
 # Señal final 
xx = S + nn
  
#%% Visualizacion de resultados

#Transformada



###Tarea: agregarle las ventanas!