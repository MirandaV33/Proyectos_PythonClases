# -- coding: utf-8 --
"""
Created on Thu Apr 10 19:07:54 2025

@author: Joaquin
"""

# -- coding: utf-8 --
"""
Created on Thu Apr  3 19:10:07 2025

@author: Joaquin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
from scipy import signal
from scipy.fft import fft, fftshift


 
N=1000
N2=10*N # Cantidad de muestras
R=200 # Realizaciones
fs=1000 #Frecuecia de muestreo
a1= np.sqrt(2) #Amplitud de la señal
df=fs/N #rResolucion espectral
df2=df*(1/10)
tt = np.arange(0,1,1/N).reshape((N,1))#Vector de tiempo de columna N
tt = np.tile(tt, (1, R))
Pn= 1/10 #Potencia de ruido cuantizado con 10 dB

omega_0=fs/4 # Frecuencia central
fr=np.random.uniform(-0.5,0.5,size=(1,R)) # Frecuencia aletearia

omega_1= omega_0 + fr*df
xx = a1*np.sin(2*np.pi*omega_1*tt) # Hay que multiplicar por 2pi sino, no queda.

sigma=np.sqrt(1/10)
nn= np.random.normal(0,sigma,size=(N,1))
nn = np.tile(nn, (1, R))

S= xx + nn # Señal de ruido 

window_1 = signal.windows.blackmanharris(N)
window_2=signal.windows.flattop(N)
window_3= signal.windows.boxcar(N)

win_1=window_1.reshape((N,1))
win_2=window_2.reshape((N,1))
win_3=window_3.reshape((N,1))

ventaneo_1 = S*win_1
ventaneo_2 = S*win_2
ventaneo_3 = S*win_3

final_fft_1=1/N*np.fft.fft(ventaneo_1,n=N2,axis=0)
final_fft_2=1/N*np.fft.fft(ventaneo_2,n=N2,axis=0)
final_fft_3=1/N*np.fft.fft(ventaneo_3,n=N2,axis=0)

final_BMH = np.abs(final_fft_1)
final_FLT= np.abs(final_fft_2)
final_BOX = np.abs(final_fft_3)

# indice= N/4
# a_gorro_1= final_BMH[250] # vector para quitar la feta
# a_gorro_2= final_FLT[250]
# a_gorro_3= final_BOX[250]
# A_GORRO = np.array([a_gorro_1, a_gorro_2, a_gorro_3])  # También da (3, 200)



k_1=np.argmax(final_BMH[:N2//2, :],axis=0)
k_2=np.argmax(final_FLT[:N2//2, :],axis=0)
k_3=np.argmax(final_BOX[:N2//2, :],axis=0)

omega_1=k_1*df2
omega_2=k_2*df2
omega_3=k_3*df2


# Etiquetas para cada conjunto
labels = ['Blackman-Harris', 'Flattop', 'Boxcar']


# Graficar los 3 histogramas superpuestos
# plt.figure(1)
# plt.hist(a_gorro_1, bins=30, label='Blackman-Harris', color='blue', alpha=0.6)
# plt.hist(a_gorro_2, bins=30, label='Flattop', color='green', alpha=0.6)
# plt.hist(a_gorro_3, bins=30, label='Boxcar', color='red', alpha=0.6)


plt.title('Histogramas de Magnitud FFT (una por ventana)')
plt.xlabel('Magnitud')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Calculo el sesgo del maximo de la señal
# esperanza_BHK=np.mean(a_gorro_1)
# esperanza_Flat=np.mean(a_gorro_2)
# esperanza_BOX=np.mean(a_gorro_3)

# valor_real=a1

# sesgo_BHK=esperanza_BHK-valor_real
# sesgo_Flat=esperanza_Flat- valor_real
# sesgo_BOX=esperanza_BOX - valor_real

# varianza_BHK=np.var(a_gorro_1)
# varianza_Flat=np.var(a_gorro_2)
# varianza_BOX=np.var(a_gorro_3)

plt.figure(2)
plt.hist(omega_1, bins=30, label='Blackman-Harris', color='purple', alpha=0.6)
plt.hist(omega_2, bins=30, label='Flattop', color='green', alpha=0.6)
plt.hist(omega_3, bins=30, label='Boxcar', color='black', alpha=0.6)

plt.title('Histogramas de omega (una por ventana)')
plt.xlabel('Magnitud')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

esperanza_BHK_omega_1=np.mean(omega_1)
esperanza_BHK_omega_1=np.mean(omega_2)
esperanza_BHK_omega_1=np.mean(omega_3)