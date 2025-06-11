# -*- coding: utf-8 -*-
"""
Created on Wed May 21 21:06:36 2025

@author: l
"""

#%% Librerias 

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio

# Funciones para filtros
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline 

#Plantilla
from pytc2.sistemas_lineales import plot_plantilla

#%% Señal ECG

# Tenemos componentes de baja frecuencias que generan interferencia! Por sistemas electricos tenemos interferencia de 50Hz 
# Tambien tenemos interferencia de alta frecuencia por el ruido muscular electrico!
# Vamos a buscar que nuestro filtro sea un  PASA BANDA, atenuando las freuencias bajas y las altas --> 0.5-40Hz

#Guardamos los datos del archivo en un diccionario de Python: enlistamos
mat_struct = sio.loadmat('./ECG_TP4.mat')

#Extraemos los datos de la señal y los metemos en un vector.
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N1 = len(ecg_one_lead)
# Normalizo la señal 
ecg_one_lead_r = ecg_one_lead / np.std(ecg_one_lead)


#%% Filtros IIR
# Metodos de aproximacion 
aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip' cower

#%% Datos de plantilla 

#Pasa Banda 
fs = 1000 # Hz
nyq_frec= fs/2
fpass = np.array([1.0, 35.0]) #Hz
ripple = 1.0 # dB
fstop = np.array([.1, 50.]) # Hz
attenuation = 40 # dB

#%% Diseño del filtro IIR

mi_sos=sig.iirdesign(fpass, fstop, ripple, attenuation, ftype=aprox_name, output= 'sos', fs=fs)
# Graficos de salida (output) tipo sos, significa que  devuelve el filtro en formato de secciones de segundo orden
#Filtro de orden 28--> 14 pares de polos y ceros.


#%% Diseño de Plantilla 

npoints = 1000

w, hh = sig.sosfreqz(mi_sos, worN=npoints) #w es los valores a los que le calculo los complejos --> hh 

w_rad = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi

w, hh = sig.sosfreqz(mi_sos, worN=w_rad)

plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(hh)+1e-15), label='mi_sos')
plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')


ax = plt.gca()
# ax.set_xlim([0, 1])
# ax.set_ylim([-60, 1])ç

plot_plantilla(filter_type = "bandpass" , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
plt.legend()
plt.show()

#%% Filtrado de la señal 

ecg_filt= sig.sosfilt(mi_sos, ecg_one_lead_r) # ---> Aplica el filtro en una sola dirección. Introduce retardo de grupo y distorsión de fase.
ecg_filt2= sig.sosfiltfilt(mi_sos, ecg_one_lead_r) #---> Aplica el filtro hacia adelante y luego hacia atrás. Esto corrige la distorsión de fase y el retardo de grupo, pero duplica la atenuación y el rizado. 
cant_muestras= len(ecg_filt)
cant_muestras2= len(ecg_filt2)


# %%Visualizacion resultados
###SEÑAL TEMPORAL###

#Sin filtrar
plt.figure() 
plt.plot(ecg_one_lead_r, label= 'Señal sin filtrar')
plt.plot(ecg_filt, label='Señal filtrada Filt')
plt.title("ECG: Electrocardiograma con ruido")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')

#Sin filtrar
plt.figure() 
plt.plot(ecg_one_lead_r[5000:12000], label= 'Señal sin filtrar')
plt.plot(ecg_filt[5000:12000], label='Señal filtrada Filt')
plt.title("ECG: Electrocardiograma con ruido")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')

plt.figure() 
plt.plot(ecg_one_lead_r, label= 'Señal sin filtrar')
plt.plot(ecg_filt2, label='Señal filtrada')
plt.title("ECG: Electrocardiograma con ruido Filt Filt ")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')

plt.show()

# ¿Como analizo el grafico? 
# La alta frecuencia se ve como "ruido" pelusa en la señal sobre ella.  
# La baja frecuencia se ve como una CONTINUA es decir, desvia la señal en amplitud! 

# Filtros FIR: Se puede desplazar, se neutaliza la demora!
#Filtros IIR: 

#IMPORTANTE:en el dominio temporal podemos observar como se comporta la envolvente! Que en el espectro es dificil de apreciar
# Este tipo de filtro, altera la morfologia de un ECG porque tiene transcisiones muy abruptas. Tenemos que bajar el Q, habria que analizar la respuesta al impulso y transcicion. Influye directamente sobre la efectividad de la banda de paso. Redefinir la plantilla. 
#Al hacer filt filt, estamos filtrando el doble de atenuacion 40-->80dB
#El rippple esta relacionado al maximo de la señal, por lo que pasa 1-->2 dB
# Filt--> Distorsion de fase 

#%% Regiones de interes: Esta parte del código te permite hacer un "zoom" en segmentos 
# específicos de la señal de ECG para poder observar de cerca el efecto de los filtros. 

regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

demora2=67
demora=0

for ii in regs_interes:

    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(16, 8), facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead_r[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ecg_filt2[zoom_region + demora], label='Filtrado')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()
regs_interes = ( 
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )

for ii in regs_interes:

    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([cant_muestras, ii[1]]), dtype='uint')
    
    plt.figure(figsize=(16, 8), facecolor='w', edgecolor='k')
    plt.plot(zoom_region, ecg_one_lead_r[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butter')
    plt.plot(zoom_region, ecg_filt2[zoom_region + demora], label='Filtrado')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
            
    plt.show()


##%% Analisis del filtro 

##-->Suma de muestras adyacentes: pasabajos
## Filtro ACUMULADOR. Estimacion de la integracion.
# x(n) + x(n-1) +x(n-2)
 
##-->Diferencia de muestras adyacentes: pasa altos- pasabandas
## Filtro diferenciador: diferencia entre puntos adyacentes obtenemos la pendiente. 
## x(n)-x(n-1)
## x(n) -x(n-2) MEJOR aproximacion a la derivacion

H_w=np.abs(hh)
H_hh=np.angle(hh)
H_hh_grapped=np.unwrap(H_hh)
##IMPORTANTE: En los filtros tenemos MUCHAS singularidades, y cada uno aporta un cacho de fase! Por lo que va alternar y caer mucho mas que `pi medios

# Módulo
plt.figure()
plt.plot(w/np.pi, H_w)
plt.title('Módulo |H(e^{{jω}})|')
plt.xlabel('ω (radianes)')
plt.ylabel('Módulo')
plt.grid(True) 
plt.show()   
 
# Fase
plt.figure()
plt.plot(w/np.pi, H_hh_grapped)
plt.title('Fase ∠H(e^{{jω}})')
plt.xlabel('ω (radianes)')
plt.ylabel('Fase (rad)')
plt.grid(True)

plt.show()

## Demora para filt: palito alto, y el proximo palito 
## Distorsion de fase: nosotros esperamos una resp de fase lineal y un retardo CONSTANTE (porque es la derivada, pendiente de la lineal)
## Cuando NO es lineal, el retarno ya no es constante y se introduce una DISTORSIONA la fase, es decir, que introduce demoras distintas para cada ancho de banda. 


#%% Filtro FIIR ----> Fase: lineal NO se distorsiona la fase. Si se quiere visualizar la señal, se corrige la demora. Sirve para ver en tiempo REAL. 

# Metodo de ventanas
freq=[0, fstop[0], fpass[0], fpass[1], fstop[1], nyq_frec]
gain=[0, 0, 1, 1, 0, 0] 

#Importante: numtamps: orden del filtro! 
#¿Por que usamos filtros de ordenes tan altos? 
Filtro_Ventana= sig.firwin2(numtaps=2505, freq=freq , gain=gain, window='hamming', fs=fs)

## Visualizacion
plt.figure()
w, h = sig.freqz(Filtro_Ventana, worN=8000, fs=fs)
plt.plot(w/np.pi*fs/2, 20 * np.log10(abs(h)))
plt.title('Respuesta en frecuencia del FIR')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.show()

# No sirve porque es muy exigente para una banda y no para la otra! 
# Entonces, necesitamos un sistema en CASCADA de un pasabajos y un pasaltos. 
# Van a ser de MENOR orden para lograr el efecto! 
# Convolucionamos dos respuestas de impulso!---> Respuesta en frecuencia desplazada multiplicacion 

# ¡A veces, diseñar un solo filtro pasabanda FIR con ventanas puede ser difícil para cumplir con todas 
# las especificaciones. Cascader un pasabajos y un pasaltos FIR puede dar más flexibilidad y un mejor
# control sobre las bandas de transición!

# Metodo de cuadrados minimos
# El método de cuadrados mínimos diseña filtros FIR minimizando el error cuadrático medio entre la 
# respuesta en frecuencia deseada y la respuesta en frecuencia del filtro diseñado. Este metodo, minimiza el error cuadratico medio por lo que en la banda de paso no encontramos problema, 
# Sin embargo, puede llevar a un mayor error en las frecuencias de las bandas alejadas de la frecuencia de corte. 
# Es un metodo no iterativo, y tiene asegurada la convergencia. 

# El rango de frecuencias es de 0 a nyq_frec (frecuencia de Nyquist). Recordar: brickwall definimos la ganancia en funcion de nuestra "caja" de paso. 
freq= [0, fstop[0], fpass[0], fpass[1], fstop[1], nyq_frec]
gain = [0, 0, 1, 1, 0, 0]

# Un orden de filtro FIR típico puede ser de 100 a 500 para empezar, puedes ajustarlo.
numtaps_ls = 2505 

# Diseño del filtro FIR 
Filtro_LS = sig.firls(numtaps_ls, freq, gain, fs=fs)

# Visualización del filtro en magnitud y fase! 
plt.figure()
w_ls, h_ls = sig.freqz(Filtro_LS, worN=8000, fs=fs)
plt.plot(w_ls/np.pi*fs/2, 20 * np.log10(abs(h_ls)))
plt.title('Respuesta en frecuencia del FIR (Cuadrados Mínimos)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid()
plt.show()

plt.figure()
plt.plot(w_ls/np.pi*fs/2, np.unwrap(np.angle(h_ls)))
plt.title('Fase del FIR (Cuadrados Mínimos)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid()
plt.show()

# Filtrado de la señal ECG con el filtro de Cuadrados Mínimos
ecg_filt_ls = sig.lfilter(Filtro_LS, 1, ecg_one_lead_r)

# Visualización de la señal filtrada en el tiempo (ajustar el zoom)
plt.figure()
plt.plot(ecg_one_lead_r[5000:12000], label='Señal sin filtrar')
# Ajustar el desfase si es necesario. Los filtros FIR introducen un retardo de (numtaps-1)/2 muestras.
delay_ls = (numtaps_ls - 1) // 2
plt.plot(ecg_filt_ls[5000 + delay_ls : 12000 + delay_ls], label='Señal filtrada (Cuadrados Mínimos)')
plt.title("ECG: Filtrado con FIR (Cuadrados Mínimos)")
plt.xlabel('Muestras')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.show()


# Metodo de Parks-McClellan (PM)


#%% Filtro No-Lineal

#Segmento de interes: 
    
ecg_segment= ecg_one_lead_r[700000:745000]
t_segment=np.linspace(0, len(ecg_one_lead_r)/ fs, len(ecg_segment))

#%% Filtro de la mediana 

# Usando ventanas conectamos filtros en cascada. Muy util para remover ruido de base de linea (impulsivo) de baja frecuencia

#Ventana
win1=201
win2=1200

# Filtro de mediana (200ms)

base_line= sig.medfilt(ecg_one_lead_r, kernel_size=win1)

# Filtro de mediana (600ms)
base_line2=sig.medfilt(base_line, kernel_size=win2)

#Visualizacion
plt.figure(figsize=(12,5)) 
plt.plot(t_segment,ecg_segment, label= 'Señal sin filtrar')
plt.plot(t_segment, base_line2, label='Linea de base - filtrado')
plt.title("ECG: Electrocardiograma filtrado con mediana")
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()


#%% Filtro por metodo de estimacion por interpolacion de splines y cubicas

# Este método es una técnica de suavizado que interpola la señal utilizando splines 
# cúbicas, que son polinomios de tercer grado que se ajustan por segmentos a la señal. 
# Se utiliza para eliminar el ruido de baja frecuencia (deriva de la línea de base) al 
# ajustar una curva suave a los puntos de la señal.

# Creamos el spline cubico con todos los puntos 
spline= CubicSpline(t_segment, ecg_segment)

# Evaluamos el spline en todos los puntos 
ecg_spline=spline(t_segment)

plt.figure(figsize=(12,5)) 
plt.plot(t_segment,ecg_segment, label= 'Señal sin filtrar')
plt.plot(t_segment, ecg_spline, label='Spline Cubico')
plt.title("ECG: Electrocardiograma filtrado con Splines Cubicos" )
plt.xlabel('Tiempo')
plt.ylabel('Amplitud [V]')
plt.legend()
plt.grid(True)
plt.show()

