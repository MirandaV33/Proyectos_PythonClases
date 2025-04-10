# %%
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 17:28:09 2025

@author: l
"""


import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

## ¿Que quiero hacer con esta tarea? 
#-----> Estimar el espectro, cuando tenemos muestras "desconocidas" 
# Para ello queremos principal encontrar tres estimadores: amplitud estimada, potencia espectral, y estimacion frecuencial

#%% Datos de la simulacion, definicion de constantes

Np=1000 
SNRs = [3, 10]
R=200 #numero de muestras
a1= np.sqrt(2) #Amplitud de la señal, eligiendo esta ya estoy normalizando la señal. Ya no hace falta dividirla por la desviacion estandar
fs = 1000 # frecuencia de muestreo (Hz) conviene numero entero conocido 
N = 1000 # cantidad de muestras
ts = 1/fs  # tiempo de muestreo
df= fs/N #resolucion espectral 
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

S= a1*np.sin(2*np.pi*omega1*tt)

# Defino la ventana de Barthann
M=N
w= signal.windows.barthann(M).reshape((Np, 1)) #[1000, 1]

# Defino la ventana de 
w2= signal.windows.blackmanharris(M).reshape((Np, 1)) #[1000, 1]

# Defino la ventana de 
w3= signal.windows.flattop(M).reshape((Np, 1)) #[1000, 1]

#Grilla de frecuencias  
freqs = np.fft.fftfreq(N, d=ts)

# %%

for snr_db in SNRs:
    # #%% Datos del ruido
    
    # # Señal analogica --> Lo saco de SNR
    pot_ruido_analog = 10**(- snr_db / 10)
    sigma= np.sqrt(pot_ruido_analog)
    # #Generacion de ruido analogico 
    nn = np.random.normal(0, sigma, (Np, R)) 
     
    # Señal final 
    xx = S + nn  # [1000x200]
    
    # Multiplicacion 
    xw= xx * w  # [1000, 200] * [1000, 1] → [1000, 200]
    xw2= xx * w2 
    xw3= xx * w3
    
    #%% Tranformada y estimadores 

    # Calculo la transformada
    ft_xx = 1/N * np.fft.fft(xx, axis=0) 
    ft_xw = 1/N * np.fft.fft(xw, axis=0) 
    ft_xw2 = 1/N * np.fft.fft(xw2, axis=0)
    ft_xw3 = 1/N * np.fft.fft(xw3, axis=0) 
    
    # Calculo el estimador de amplitud (a1=mod(ft_xw))
    
    #Estimador---> en N/4 de la matriz xx  es 
    
    
    # # Estimador sin ventanear
    a1_est= np.abs(ft_xx[N//4, :])  # [1000, 200] // Division entera
    # a1_est_max = np.max(a1_est, axis=0) 
    
    
    # # Estimador ventana barthann
    a1_est2= np.abs(ft_xw[N//4, : ])  # [1000, 200]
    # a1_est_max2 = np.max(a1_est2, axis=0) 
    
    # # Estimador ventana blackmanharris
    a1_est3= np.abs(ft_xw2[N//4, :])  # [1000, 200]
    # a1_est_max3 = np.max(a1_est3, axis=0) 
    
    # # Estimador  ventana flattop
    a1_est4= np.abs(ft_xw3[N//4, :])  # [1000, 200]
    # a1_est_max4 = np.max(a1_est4, axis=0) 
    
    #Calculo el estimador frecuencial sin ventana
    # Calculo el estimador de potencia P=1/N*mod[ft_xw]^2
    X_xxabs= np.abs(ft_xx)
    P_est= 1/N * (X_xxabs)**2  # [1000, 200]
    omega1_est= np.argmax(P_est, axis=0)  # [200] → una potencia maxima para una frecuencia por realización. 

    
    #Calculo del estimador frecuencial de brithann 
    # Calculo el estimador de potencia P=1/N*mod[ft_xw]^2
    X_xwabs1= np.abs(ft_xw)
    P_est1= 1/N * (X_xwabs1)**2  # [1000, 200]
    omega2_est= np.argmax(P_est1, axis=0)  # [200] → una potencia maxima para una frecuencia por realización. 
    
    
    #Calculo el estimador frecuencial de blackmanharris
    # Calculo el estimador de potencia P=1/N*mod[ft_xw]^2
    X_xwabs2= np.abs(ft_xw2)
    P_est2= 1/N * (X_xwabs2)**2  # [1000, 200]
    omega3_est= np.argmax(P_est2, axis=0)  # [200] → una potencia maxima para una frecuencia por realización. //Argumento que MAXIMIZA el modulo de la transformada 
    
    
    #Calculo el estimador frecuencial de flattop
    # Calculo el estimador de potencia P=1/N*mod[ft_xw]^2
    X_xwabs3= np.abs(ft_xw3)
    P_est3= 1/N * (X_xwabs3)**2  # [1000, 200]
    omega4_est = np.argmax(P_est3, axis=0)  # [200] → una potencia maxima para una frecuencia por realización. //Axis=0 hace que  
    
    #Calculo el sesgo 
    
    
    
    ###HISTOGRAMA###
    plt.figure()
    plt.hist(omega1_est, bins=10, color='red', alpha=0.5, label="Estimador sin ventanear")
    plt.hist(omega2_est, bins=10, color='green',alpha=0.5, label="Estimador ventana barthann")
    plt.hist(omega3_est, bins=10, color='blue',alpha=0.5, label="Estimador ventana blackmanharris")
    plt.hist(omega4_est, bins=10, color='pink', alpha=0.5, label="Estimador ventana flattop")
    plt.title("Histograma de frecuencias estimadas (200 realizaciones)- SNR {} dB".format(snr_db))
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Cantidad de ocurrencias")
    plt.grid(True)
    plt.legend()
    
    plt.figure()
    plt.hist(a1_est, bins=10, color='red', alpha=0.5, label="Estimador sin ventanear") #Bins: resolucion espectral del histograma; conteo relativo. ANCHURA de los valores.
    plt.hist(a1_est2, bins=10, color='green', alpha=0.5, label="Estimador ventana barthann")
    plt.hist(a1_est3, bins=10, color='blue', alpha=0.5, label="Estimador ventana blackmanharris")
    plt.hist(a1_est4, bins=10, color='pink', alpha=0.5, label="Estimador ventana flattop")
    plt.legend()

    plt.title("Histograma de amplitudes estimadas - SNR {} dB".format(snr_db))
    plt.xlabel("Amplitud estimada")
    plt.ylabel("Cantidad de ocurrencias")
    plt.grid(True)

    plt.figure()
    bfrec = ff <= fs/2 #Vector de n valores que hace que me quede con LA MITAD el vector, nos devuelve un verdadero o falso segun la condicion 
    
    plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xx[bfrec])**2))
    # plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xw[bfrec])**2), label="Señal con ventana brathamm")
    #plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xw2[bfrec])**2), label="Señal con ventana blackmanharris")
    # plt.plot(ff[bfrec], 10 * np.log10(2 * np.abs(ft_xw3[bfrec])**2), label="Señal con ventana flattop")
    # plt.title('Representacion espectral' )
    plt.ylabel('Densidad de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    
plt.show()


