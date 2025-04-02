#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:58:13 2025

@author: mariano
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
tt, xx = mi_funcion_sen(2, 0, 1.5, 0, 1000, 1000)

# Observamos que varianza tiene
print(f"Varianza: {np.var(xx)}")
# Observamos el desvío estándar
print(f"Desvío estándar: {np.std(xx)}")
# Normalizamos la función con el desvío estándar
xn = xx / np.std(xx)
print(xn)


# # Graficamos la señal normalizada
# plt.figure(1)
# plt.plot(tt, xn, color='b')
# plt.title('Señal Normalizada')
# plt.xlabel('Tiempo [segundos]')
# plt.ylabel('Amplitud')
# plt.grid(True)
# plt.show()

    # Generar un ruido el cual tengo magnitud de 50nV/(raizHz)
    # Escalarla en funcion de la potencia del datasheet
    
    
#%% Datos de la simulación

fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
#Normalizamos la resolucion espectral. 

# Datos del ADC
B = 8  # bits
Vf = 1.5 # rango simétrico de +/- Vf Volts
q =  Vf/2**(B-1) # paso de cuantización de q Volts

##Output noise: sonidodel data sheet que nos sirve para inducirle a la señal 


# datos del ruido (potencia de la señal normalizada, es decir 1 W). Nos interesa que la potencia sea 1W porque en dB es mas facil de medir. 
#Genera una distribucion de puntos con desviacion estandar normalizada, igual a 1. 
pot_ruido_cuant = (q**(2))/12  # Watts  --> es en realidad la VARIANZA del error de cuantizacion
#!!! Potencia de ruido= varianza de cuantizacion!
kn = 1. # escala de la potencia de ruido analógico
#Relacion ruido analogico y de cuantizacion, ajusta cuánta potencia de ruido se considera en la señal analógica antes de la digitalización.
pot_ruido_analog = pot_ruido_cuant * kn 

#Si el ruido analógico es muy alto, el ruido de cuantización tiene un impacto menor en el resultado final,
#ya que el ruido analógico puede "dominar" sobre el ruido de cuantización. Si el ruido analógico es bajo o despreciable, 
#el ruido de cuantización será más notable y afectará más a la calidad de la señal digitalizada.

#Aca podemos hablar del PISO DE RUIDO. El piso de ruido (o ruido de fondo) es el nivel mínimo de señal de ruido que se 
#puede detectar en un sistema de medición. Generalmente esta relacionado con el piso analogico, ya que TODA señal analogica tiene
# asociado algun grado de ruido de diferentes fuentes. 
#La potencia de ruido se puede entender como una medida de la cantidad de ruido presente en un sistema, 
#y su relación con el piso de ruido se da a través de la densidad espectral de potencia. El piso de ruido está
#relacionado con la potencia de ruido total en el sistema, y si el ruido analógico tiene mayor potencia que el
# ruido de cuantización, entonces el impacto del ruido de cuantización en la señal digitalizada será menor.

#Si en el modelo que estás utilizando  kn (que es el factor de escala de la potencia del ruido analógico) es grande, 
#eso significa que el ruido analógico tiene más potencia y puede dominar sobre el ruido de cuantización. En cambio, si kn es pequeño (casi 0), el ruido analógico es muy bajo, y el ruido de cuantización será más importante.

ts = 1/fs # tiempo de muestreo
df =  fs/N # resolución espectral





#%% Experimento: 
"""
   Se desea simular el efecto de la cuantización sobre una señal senoidal de 
   frecuencia 1 Hz. La señal "analógica" podría tener añadida una cantidad de 
   ruido gausiano e incorrelado.
   
   Se pide analizar el efecto del muestreo y cuantización sobre la señal 
   analógica. Para ello se proponen una serie de gráficas que tendrá que ayudar
   a construir para luego analizar los resultados.
   
"""
#np.random.normal
#np.random.uniform


# Señales

analog_sig = xx/np.std(xx)# señal analógica sin ruido
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N) # señal de ruido de analógico
sr = analog_sig + nn # señal analógica de entrada al ADC (con ruido analógico)

#Divido por q, redondeo y después multiplico por q. 
srq = np.round(sr/q)*q # señal cuantizada
# Esto o que hace es dividir la señal en q escalones, y luego redondea cada punto a un escalon q cercano. Este proceso comprende el error de redondeo lo cual nos da una señal con RUIDO


#Parametros: sr/q valores de entrada, 0 para que sean 0 decimales, y N valores de salida
nq =  srq -sr # señal de ruido de cuantización, señal cuantizada menos el ruido de cuantizacion

# plt.figure(2)
# plt.plot(tt, srq)
# plt.title('Señal Normalizada')
# plt.xlabel('Tiempo [segundos]')
# plt.ylabel('Amplitud')
# plt.grid(True)
# plt.show()


# plt.figure(3)
# plt.plot(tt,nq) 
# plt.title("Señal analogica con ruido:")
# plt.xlabel("tiempo [segundos]")
# plt.ylabel("Amplitud")
# plt.legend()
# plt.show()


#%% Visualización de resultados

# # cierro ventanas anteriores
# plt.close('all')

##################
# Señal temporal
##################

# ¿Que pasa si tengo una frecuencia de muestreo de 1hz y N muertras cual es la resolucion espectral? 
# df= 1Hz es ENTERO o seria un multiplo de df
# ¿Que pasa si no le damos una frecuencia multiplo de df? ¿Si la frecuencia de la señal no coincide con la resoluncion? 
# SE DISPERSA LA ENERGIA, el espectro (delta) no va a concentrarse tanto. 


plt.figure(1)

plt.plot(tt, srq, lw=2, linestyle='', color='blue', marker='o', markersize=5, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label='ADC out (diezmada)')
plt.plot(tt, sr, lw=1, color='black', marker='x', ls='dotted', label='$ s $ (analog)')

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########

plt.figure(2)
ft_SR = 1/N*np.fft.fft( sr) 
ft_Srq = 1/N*np.fft.fft( srq)
ft_As = 1/N*np.fft.fft( analog_sig)
ft_Nq = 1/N*np.fft.fft( nq)
ft_Nn = 1/N*np.fft.fft( nn)
#Graficamos de 0 a nyquist, porque sabemos que la transformada tiene simetria en 0. 

# grilla de sampleo frecuencial
ff = np.linspace(0, (N-1)*df, N)

bfrec = ff <= fs/2 #Vector de n valores que hace que me quede con LA MITAD el vector, nos devuelve un verdadero o falso segun la condicion 

Nnq_mean = np.mean(np.abs(ft_Nq)**2)
nNn_mean = np.mean(np.abs(ft_Nn)**2)

plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_As[bfrec])**2), color='orange', ls='dotted', label='$ s $ (sig.)' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB'.format(10* np.log10(2* nNn_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $' )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\}$' )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB'.format(10* np.log10(2* Nnq_mean)) )
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nn[bfrec])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
plt.plot( np.array([ ff[bfrec][-1], ff[bfrec][-1] ]), plt.ylim(), ':k', label='BW', lw = 0.5  )

plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()

#############
# Histograma
#############

#El histrograma tiene como proposito mostrar la distribucion de los pasos de cuantizacion. 

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')
