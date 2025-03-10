# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 19:50:11 2025

@author: l
"""

import numpy as np 
import matplotlib.pyplot as plt 

fs=1000 ##Frecuencia e muestreo
f0=50 ##Frecuencia 
N=1000 ##Cantidad de muestras 

ts= 1/fs ##Tiempo de muestreo
ds= fs/N ##Resolucion espectral 

##Grilla de sampleo temporal
tt= np.linspace(0, (N-1)*ts, N)

##Grilla de sampleo en x (amplitud)
xx= np.sin(2*np.pi*f0*tt)

plt.figure(1)
line_hdls = plt.plot(tt, xx)
plt.title('Señal: ')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud')

##Cuando f0 es 1, vamos a tener por defecto MUCHAS muestras por segundo --->1000
## fs/f0=muestrasporciclo
##Cuando f0 es 400, esta muy cerca de fs/2 por lo que el comportamiento es ERRATICO. 

#%% Preguntas y respuestas de clase
##¿Cuando y por qué da una linea plana?

##¿Que pasa cuando vamos cambiando f0?

##¿Que pasa cuando desplazamos la senoidal en pi/2?


