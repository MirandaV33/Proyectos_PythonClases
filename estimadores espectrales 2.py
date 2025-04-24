# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 19:04:49 2025

@author: l
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

Np=1000 
SNR = 10
R=200 #numero de muestras
a2= np.sqrt(2) #Amplitud de la señal, eligiendo esta ya estoy normalizando la señal. Ya no hace falta dividirla por la desviacion estandar
fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
N2= 10*N
ts = 1/fs  # tiempo de muestreo
df= fs/N #resolucion espectral 
df_pp=fs/N2
omega0= fs/4

#%% Generacion de la señal X

# Defino S

# Grilla de sampleo temporal ---> DISRETIZACION DEL TIEMPO (muestreo)
tt = np.linspace(0, (N-1)*ts, N).reshape((1000, 1))  #[1000x1]
tt= np.tile(tt, (1, R)) #Repetidor [100x200]

# Grilla sampleo frecuencial
ff= np.linspace(0, (N-1)*df, N) #.reshape(1, 1000) # [1,1000]
fr = np.random.uniform(-1/2, 1/2, size=(1,R)) # [1, 200]

omega1= omega0 + fr* (df)

S= a2*np.sin(2*np.pi*omega1*tt)

#Grilla de frecuencias  
freqs = np.fft.fftfreq(N, d=ts)

# # Señal analogica --> Lo saco de SNR
pot_ruido_analog = 10**(- SNR / 10)
sigma= np.sqrt(pot_ruido_analog)
# #Generacion de ruido analogico 
nn = np.random.normal(0, sigma, (Np, R)) 
     
# Señal final 
xx = S + nn  # [1000x200]
    
#Estimador welch 
#F tiene un tamaño de 126 porquees la mitad de 250

f,Pxx=signal.welch(xx,fs,nfft=N,window='flattop',nperseg=N/2,axis=0)
plt.figure(1)
plt.plot(f, 10* np.log10(2*np.abs(Pxx)**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density with 10 dB Noise')
plt.show()


f_3,Pxx_3=signal.welch(xx,fs,nfft=N,window='flattop',nperseg=N/16,axis=0)
#Nperseg= largo de los segmentos para superponer/autocorrelar (noverloap)
plt.figure(2)
plt.plot(f_3, 10* np.log10(2*np.abs(Pxx_3)**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density with 10 dB Noise')
plt.show()

f_4,Pxx_4=signal.welch(xx,fs,nfft=N,window='flattop',nperseg=N/4,axis=0)
plt.figure(3)
plt.plot(f_4, 10* np.log10(2*np.abs(Pxx_4)**2))
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density with 10 dB Noise')
plt.show()

#Me quedo con la feta de 250 (maximo)
a2_max1 = np.max(Pxx, axis=0) 
a2_max2 = np.max(Pxx_3, axis=0) 
a2_max3 = np.max(Pxx_4, axis=0) 


##Histograma
plt.figure()
plt.hist(a2_max1, bins=30, color='red', alpha=0.5, label="Estimador sin ventanear")
plt.hist(a2_max2, bins=30, color='red', alpha=0.5, label="Estimador sin ventanear")
plt.hist(a2_max3, bins=30, color='red', alpha=0.5, label="Estimador sin ventanear")
plt.title("Histograma de frecuencias estimadas (200 realizaciones)")
plt.xlabel("Amplitud")
plt.ylabel("Cantidad de ocurrencias")
plt.grid(True)
plt.legend()

#%% Analizo varianza y sesgo

#N/2
valor_real= a2
esperanza_a2_max = np.mean(a2_max1)
sesgo_a2_est= esperanza_a2_max-valor_real
varianza_a2_max= np.var(a2_max1)

#N/16
valor_real= a2
esperanza_a2_max = np.mean(a2_max2)
sesgo_a2_est= esperanza_a2_max-valor_real
varianza_a2_max= np.var(a2_max2)

#N/4
valor_real= a2
esperanza_a2_max = np.mean(a2_max3)
sesgo_a2_est= esperanza_a2_max-valor_real
varianza_a2_max= np.var(a2_max3)

 #%% Blackman -Tukey Method 
 
 
 
 
