# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 11:33:38 2025

@author: l
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

#%% ADC 4BITS
#%% Funciones a importar

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    
    ts = 1/fs  # tiempo de muestreo
    
    # grilla de sampleo temporal ---> DISRETIZACION DEL TIEMPO (muestreo)
    tt = np.linspace(0, (nn-1)*ts, nn)  # Cambié N por nn
    
    # Grilla de amplitud
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc  # Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx
#%% Datos de la simulacion 

#Datos para el muestreo (sampleo)
fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
 

#Proceso de CUANTIFICACION
# Datos del ADC
B = 4  # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/2**(B-1) # paso de cuantización de q Volts

#Generacion de ruido
pot_ruido_cuant = (q**(2))/12  # Watts 
kn = 1 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn 

ts = 1/fs # tiempo de muestreo
df =  fs/N # resolución espectral

# %% Llamada a la función y procesamiento
tt, xx = mi_funcion_sen(vmax=2, dc=0, ff=1, ph=0, nn=N, fs=fs)

# Normalizamos la función con el desvío estándar
xn = xx / np.std(xx)
print(xn)
#De esta manera, la varianza es unitaria, y la potencia de 1W

#%% Experimento 

# Señal analogica
analog_sig = xn

#Generacion de ruido analogico aleatorio
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)

#Señal analogica con ruido 
sr = analog_sig + nn 


#Señal cuantizada con ruido
srq = np.round(sr/q)*q 

#Ruido de cuantizacion
nq =  srq -sr 


#%% Visualiacion de resultados 

##################
# Señal temporal
##################

plt.figure(1)


#Grafico de la señal sin ruido
plt.plot(tt, analog_sig, lw=1,linestyle='-', color ="red", label='$ s $ = señal analogica' )
# Gráfico de señal cuantizada con ruido
plt.plot(tt, srq, lw=2, linestyle='-', color='blue', marker='o', markersize=3, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label=' $ srq $ = señal cuantizada')
# Gráfico de señal analógica con ruido
plt.plot(tt, sr, lw=1, color='black', marker='x', linestyle='dotted', markersize=2, markerfacecolor='red', markeredgecolor='black', fillstyle='none', label='$ sr $ =  señal analogica con ruido')


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

plt.figure(3)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

#%% ADC 4 BITS KN=1/10
#%% Funciones a importar

# Senoidal cuya potencia tenga 1Watt-> Esto facilita la conversion en dB

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    
    ts = 1/fs  # tiempo de muestreo
    
    # grilla de sampleo temporal ---> DISRETIZACION DEL TIEMPO (muestreo)
    tt = np.linspace(0, (nn-1)*ts, nn)  # Cambié N por nn
    
    # Grilla de amplitud
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc  # Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx
#%% Datos de la simulacion 


#Datos para el muestreo (sampleo)
fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
 

#Proceso de CUANTIFICACION
# Datos del ADC
B = 4  # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/2**(B-1) # paso de cuantización de q Volts

#Generacion de ruido
pot_ruido_cuant = (q**(2))/12  # Watts 
kn = 1/10 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn 

ts = 1/fs # tiempo de muestreo
df =  fs/N # resolución espectral

# %% Llamada a la función y procesamiento
tt, xx = mi_funcion_sen(vmax=2, dc=0, ff=1, ph=0, nn=N, fs=fs)

# Normalizamos la función con el desvío estándar
xn = xx / np.std(xx)
print(xn)
#De esta manera, la varianza es unitaria, y la potencia de 1W

#%% Experimento 

# Señal analogica
analog_sig = xn

#Generacion de ruido analogico aleatorio
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)

#Señal analogica con ruido 
sr = analog_sig + nn 


#Señal cuantizada con ruido
srq = np.round(sr/q)*q 

#Ruido de cuantizacion
nq =  srq -sr 


#%% Visualiacion de resultados 

##################
# Señal temporal
##################

plt.figure(4)


#Grafico de la señal sin ruido
plt.plot(tt, analog_sig, lw=1,linestyle='-', color ="red", label='$ s $ = señal analogica' )
# Gráfico de señal cuantizada con ruido
plt.plot(tt, srq, lw=2, linestyle='-', color='blue', marker='o', markersize=3, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label=' $ srq $ = señal cuantizada')
# Gráfico de señal analógica con ruido
plt.plot(tt, sr, lw=1, color='black', marker='x', linestyle='dotted', markersize=2, markerfacecolor='red', markeredgecolor='black', fillstyle='none', label='$ sr $ =  señal analogica con ruido')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########

plt.figure(5)

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

plt.figure(6)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

#%% ADC 8 BITS KN=1
#%% Funciones a importar

# Senoidal cuya potencia tenga 1Watt-> Esto facilita la conversion en dB

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    
    ts = 1/fs  # tiempo de muestreo
    
    # grilla de sampleo temporal ---> DISRETIZACION DEL TIEMPO (muestreo)
    tt = np.linspace(0, (nn-1)*ts, nn)  # Cambié N por nn
    
    # Grilla de amplitud
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc  # Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx
#%% Datos de la simulacion 

#Datos para el muestreo (sampleo)
fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
 

#Proceso de CUANTIFICACION
# Datos del ADC
B = 8  # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/2**(B-1) # paso de cuantización de q Volts

#Generacion de ruido
pot_ruido_cuant = (q**(2))/12  # Watts 
kn = 1. # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn 

ts = 1/fs # tiempo de muestreo
df =  fs/N # resolución espectral

# %% Llamada a la función y procesamiento
tt, xx = mi_funcion_sen(vmax=2, dc=0, ff=1, ph=0, nn=N, fs=fs)

# Normalizamos la función con el desvío estándar
xn = xx / np.std(xx)
print(xn)
#De esta manera, la varianza es unitaria, y la potencia de 1W

#%% Experimento 

# Señal analogica
analog_sig = xn

#Generacion de ruido analogico aleatorio
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)

#Señal analogica con ruido 
sr = analog_sig + nn 


#Señal cuantizada con ruido
srq = np.round(sr/q)*q 

#Ruido de cuantizacion
nq =  srq -sr 


#%% Visualiacion de resultados 

##################
# Señal temporal
##################

plt.figure(7)


#Grafico de la señal sin ruido
plt.plot(tt, analog_sig, lw=1,linestyle='-', color ="red", label='$ s $ = señal analogica' )
# Gráfico de señal cuantizada con ruido
plt.plot(tt, srq, lw=2, linestyle='-', color='blue', marker='o', markersize=3, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label=' $ srq $ = señal cuantizada')
# Gráfico de señal analógica con ruido
plt.plot(tt, sr, lw=1, color='black', marker='x', linestyle='dotted', markersize=2, markerfacecolor='red', markeredgecolor='black', fillstyle='none', label='$ sr $ =  señal analogica con ruido')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########

plt.figure(8)

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

plt.figure(9)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')


#%% ADC 16 BITS KN=10

#%% Funciones a importar

# Senoidal cuya potencia tenga 1Watt-> Esto facilita la conversion en dB

def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
    
    ts = 1/fs  # tiempo de muestreo
    
    # grilla de sampleo temporal ---> DISRETIZACION DEL TIEMPO (muestreo)
    tt = np.linspace(0, (nn-1)*ts, nn)  # Cambié N por nn
    
    # Grilla de amplitud
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc  # Sen (W0 . t + fase) // W0 = 2pi . t 
    
    return tt, xx
#%% Datos de la simulacion 

#Datos para el muestreo (sampleo)
fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
 

#Proceso de CUANTIFICACION
# Datos del ADC
B = 16  # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/2**(B-1) # paso de cuantización de q Volts

#Generacion de ruido
pot_ruido_cuant = (q**(2))/12  # Watts 
kn = 10 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn 

ts = 1/fs # tiempo de muestreo
df =  fs/N # resolución espectral

# %% Llamada a la función y procesamiento
tt, xx = mi_funcion_sen(vmax=2, dc=0, ff=1, ph=0, nn=N, fs=fs)

# Normalizamos la función con el desvío estándar
xn = xx / np.std(xx)
print(xn)
#De esta manera, la varianza es unitaria, y la potencia de 1W

#%% Experimento 

# Señal analogica
analog_sig = xn

#Generacion de ruido analogico aleatorio
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)

#Señal analogica con ruido 
sr = analog_sig + nn 


#Señal cuantizada con ruido
srq = np.round(sr/q)*q 

#Ruido de cuantizacion
nq =  srq -sr 


#%% Visualiacion de resultados 

##################
# Señal temporal
##################

plt.figure(10)


#Grafico de la señal sin ruido
plt.plot(tt, analog_sig, lw=1,linestyle='-', color ="red", label='$ s $ = señal analogica' )
# Gráfico de señal cuantizada con ruido
plt.plot(tt, srq, lw=2, linestyle='-', color='blue', marker='o', markersize=3, markerfacecolor='blue', markeredgecolor='blue', fillstyle='none', label=' $ srq $ = señal cuantizada')
# Gráfico de señal analógica con ruido
plt.plot(tt, sr, lw=1, color='black', marker='x', linestyle='dotted', markersize=2, markerfacecolor='red', markeredgecolor='black', fillstyle='none', label='$ sr $ =  señal analogica con ruido')


plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()


###########
# Espectro
###########

plt.figure(11)

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

plt.figure(12)
bins = 10
plt.hist(nq.flatten()/(q), bins=bins)
plt.plot( np.array([-1/2, -1/2, 1/2, 1/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))

plt.xlabel('Pasos de cuantización (q) [V]')

#%% Simular elaliasing 
#%% Datos de la simulacion 

#Datos para el muestreo (sampleo)
fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
 

#Proceso de CUANTIFICACION
# Datos del ADC
B = 4  # bits
Vf = 2 # rango simétrico de +/- Vf Volts
q =  Vf/2**(B-1) # paso de cuantización de q Volts

#Generacion de ruido
pot_ruido_cuant = (q**(2))/12  # Watts 
kn = 1 # escala de la potencia de ruido analógico
pot_ruido_analog = pot_ruido_cuant * kn 

ts = 1/fs # tiempo de muestreo
df =  fs/N # resolución espectral

# %% Llamada a la función y procesamiento
tt, xx = mi_funcion_sen(vmax=2, dc=0, ff=500, ph=0, nn=N, fs=fs)

# Normalizamos la función con el desvío estándar
xn = xx / np.std(xx)
print(xn)
#De esta manera, la varianza es unitaria, y la potencia de 1W

#%% Experimento 

# Señal analogica
analog_sig = xn

#Generacion de ruido analogico aleatorio
nn = np.random.normal(0, np.sqrt(pot_ruido_analog), N)

#Señal analogica con ruido 
sr = analog_sig + nn 


#Señal cuantizada con ruido
srq = np.round(sr/q)*q 

#Ruido de cuantizacion
nq =  srq -sr 

#%% Graficar los resultados

plt.figure(14)

# Señales 
plt.plot(tt, sr, label='Señal analógica con ruido', color= 'black', linewidth=1)
plt.vlines(tt, ymin=0, ymax=srq, colors='r', linestyles='dotted', label='Muestreo')

# Títulos y etiquetas
plt.title('Simulación de Aliasing y muestreo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()
