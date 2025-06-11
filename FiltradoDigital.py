# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 20:33:20 2025

@author: l
"""
 
#%% Librerias 

import sympy as sp
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from pytc2.sistemas_lineales import plot_plantilla

#%% Definicion de funciones 

#Funcion para comparar filtros 

def plot_freq_resp_fir(this_num, this_desc):

    wrad, hh = sig.freqz(this_num, 1.0)
    ww = wrad / np.pi
    
    plt.figure(1)

    plt.plot(ww, 20 * np.log10(abs(hh)), label=this_desc)

    plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef) )
    plt.xlabel('Frequencia normalizada')
    plt.ylabel('Modulo [dB]')
    plt.grid(which='both', axis='both')

    axes_hdl = plt.gca()
    axes_hdl.legend()
    
    plt.figure(2)

    phase = np.unwrap(np.angle(hh))

    plt.plot(ww, phase, label=this_desc)

    plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef))
    plt.xlabel('Frequencia normalizada')
    plt.ylabel('Fase [rad]')
    plt.grid(which='both', axis='both')

    axes_hdl = plt.gca()
    axes_hdl.legend()

    plt.figure(3)

    # ojo al escalar Omega y luego calcular la derivada.
    gd_win = group_delay(wrad, phase)

    plt.plot(ww, gd_win, label=this_desc)

    plt.ylim((np.min(gd_win[2:-2])-1, np.max(gd_win[2:-2])+1))
    plt.title('FIR diseñado por métodos directos - Taps:' + str(cant_coef))
    plt.xlabel('Frequencia normalizada')
    plt.ylabel('Retardo [# muestras]')
    plt.grid(which='both', axis='both')

    axes_hdl = plt.gca()
    axes_hdl.legend()    

plot_freq_resp_fir(num_bh, filter_type+ '-blackmanharris')    
plot_freq_resp_fir(num_hm, filter_type+ '-hamming')    
plot_freq_resp_fir(num_ka, filter_type+ '-kaiser-b14')    
    
    
# sobreimprimimos la plantilla del filtro requerido para mejorar la visualización    
fig = plt.figure(1)    
plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
ax = plt.gca()
ax.legend()

# reordenamos las figuras en el orden habitual: módulo-fase-retardo
plt.figure(2)    
axes_hdl = plt.gca()
axes_hdl.legend()

plt.figure(3)    
axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

#%% Filtros IIR
# Metodos de aproximacion 

aprox_name = 'butter'
# aprox_name = 'cheby1'
# aprox_name = 'cheby2'
# aprox_name = 'ellip' cower
#aprox_name = 'bessel'

# Diseño de plantilla 

filter_type = 'lowpass'
# filter_type = 'highpass'
# filter_type = 'bandpass'
# filter_type = 'bandstop'


# plantillas normalizadas a Nyquist y en dB

if filter_type == 'lowpass':

    # fpass = 1/2/np.pi # 
    fpass = 0.25 # 
    ripple = 0.5 # dB
    fstop = 0.6 # Hz
    attenuation = 40 # dB

elif filter_type == 'highpass':

    fpass = 0.6 
    ripple = 0.5 # dB
    fstop = 0.25
    attenuation = 40 # dB

elif filter_type == 'bandpass':

    fpass = np.array( [0.4, 0.6] ) 
    ripple = 0.5 # dB
    fstop = np.array( [0.25, 0.75] ) 
    attenuation = 40 # dB
    
else:

    # bandstop
    fpass = np.array( [0.25, 0.75] ) 
    ripple = 0.5 # dB
    fstop = np.array( [0.4, 0.6] ) 
    attenuation = 40 # dB


    # Cálculo del filtro

# frecuencia de muestreo normalizada (Nyquist = 1)
fs = 2

if aprox_name == 'butter':

    order, wcutof = sig.buttord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.buttord( fpass, fstop, ripple, attenuation, analog=False)

elif aprox_name == 'cheby1':

    order, wcutof = sig.cheb1ord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.cheb1ord( fpass, fstop, ripple, attenuation, analog=False)
    
elif aprox_name == 'cheby2':

    order, wcutof = sig.cheb2ord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.cheb2ord( fpass, fstop, ripple, attenuation, analog=False)
    
elif aprox_name == 'ellip':
   
    order, wcutof = sig.ellipord( 2*np.pi*fpass*fs/2, 2*np.pi*fstop*fs/2, ripple, attenuation, analog=True)
    orderz, wcutofz = sig.ellipord( fpass, fstop, ripple, attenuation, analog=False)


# Diseño del filtro analógico

num, den = sig.iirfilter(order, wcutof, rp=ripple, rs=attenuation, btype=filter_type, analog=True, ftype=aprox_name)

my_analog_filter = sig.TransferFunction(num,den)
my_analog_filter_desc = aprox_name + '_ord_' + str(order) + '_analog'

# Diseño del filtro digital

numz, denz = sig.iirfilter(orderz, wcutofz, rp=ripple, rs=attenuation, btype=filter_type, analog=False, ftype=aprox_name)

my_digital_filter = sig.TransferFunction(numz, denz, dt=1/fs)
my_digital_filter_desc = aprox_name + '_ord_' + str(orderz) + '_digital'



# Plantilla de diseño

plt.figure(1)
plt.cla()

npoints = 1000
w_nyq = 2*np.pi*fs/2

w, mag, _ = my_analog_filter.bode(npoints)
plt.plot(w/w_nyq, mag, label=my_analog_filter_desc)

w, mag, _ = my_digital_filter.bode(npoints)
plt.plot(w/w_nyq, mag, label=my_digital_filter_desc)

plt.title('Plantilla de diseño')
plt.xlabel('Frecuencia normalizada a Nyq [#]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', axis='both')

plt.gca().set_xlim([0, 1])

plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)



#%% Filtros FIR
#%% Metodo de ventanas para filtro pasa bajos 

# frecuencia de muestreo normalizada
fs = 2.0
# tamaño de la respuesta al impulso
cant_coef = 27

filter_type = 'lowpass'

fpass = 0.25 # 
ripple = 0.5 # dB
fstop = 0.6 # Hz
attenuation = 40 # dB

# construyo la plantilla de requerimientos
frecs = [0.0,  fpass,     fstop,          1.0]
gains = [0,   -ripple, -attenuation,   -np.inf] # dB

gains = 10**(np.array(gains)/20)

# algunas ventanas para evaluar
#win_name = 'boxcar'
#win_name = 
win_name = 'kaiser'
#win_name = 'flattop'

# FIR design
num_bh = sig.firwin2(cant_coef, frecs, gains , window='blackmanharris' )
num_hm = sig.firwin2(cant_coef, frecs, gains , window='hamming' )
num_ka = sig.firwin2(cant_coef, frecs, gains , window=('kaiser',14))
den = 1.0

#%% Cuadrados minimos 

# frecuencia de muestreo normalizada
fs = 2.0
# tamaño de la respuesta al impulso
cant_coef = 27

filter_type = 'lowpass'

fpass = 0.25 # 
ripple = 0.5 # dB
fstop = 0.6 # Hz
attenuation = 40 # dB

# construyo la plantilla de requerimientos
frecs = [0.0,  fpass,     fstop,          1.0]
gains = [0,   -ripple, -attenuation,   -np.inf] # dB

gains = 10**(np.array(gains)/20)

num_firls = sig.firls(cant_coef, frecs, gains, fs=fs)
num_hm = sig.firwin2(cant_coef, frecs, gains , window='hamming' )


plot_freq_resp_fir(num_firls, filter_type + '-firls')    
plot_freq_resp_fir(num_hm, filter_type + '-hamming')    
  
# sobreimprimimos la plantilla del filtro requerido para mejorar la visualización    
plt.figure(1)    
plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
axes_hdl = plt.gca()
axes_hdl.legend()

# reordenamos las figuras en el orden habitual: módulo-fase-retardo
plt.figure(2)    
axes_hdl = plt.gca()
axes_hdl.legend()

plt.figure(3)    
axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()

#%% Parks-McClellan (PM)

num_firls = sig.firls(cant_coef, frecs, gains, fs=fs)
num_hm = sig.firwin2(cant_coef, frecs, gains , window='hamming' )
num_remez = sig.remez(cant_coef, frecs, gains[::2], fs=fs)


plot_freq_resp_fir(num_remez, filter_type + '-remez')    
plot_freq_resp_fir(num_firls, filter_type + '-firls')    
plot_freq_resp_fir(num_hm, filter_type + '-hamming')    
  
# sobreimprimimos la plantilla del filtro requerido para mejorar la visualización    
plt.figure(1)    
plot_plantilla(filter_type = filter_type , fpass = fpass, ripple = ripple , fstop = fstop, attenuation = attenuation, fs = fs)
axes_hdl = plt.gca()
axes_hdl.legend()

# reordenamos las figuras en el orden habitual: módulo-fase-retardo
plt.figure()    
axes_hdl = plt.gca()
axes_hdl.legend()

plt.figure()    
axes_hdl = plt.gca()
axes_hdl.legend()

plt.show()