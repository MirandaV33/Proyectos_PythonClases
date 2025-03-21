# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:38:39 2025

@author: l
"""


#%% Librerias 

import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as signal

#%% Datos a simular

L = 1 # Inductancia en Henrios
C = 1 # Capacitancia en Faradios
R = 1 # Resistencia en Ohmios
 

#%%  Función de Transferencia Filtro 1

numerador = [R / L, 0]
denominador = [1, R / L, 1 / (L * C)]

# Crear el sistema en el dominio de Laplace 
sistema = signal.TransferFunction(numerador, denominador)

# Calcular los polos y ceros del sistema
zeros, poles, ganancia = signal.tf2zpk(numerador, denominador)

# Calcular la frecuencia de resonancia w0
w_0 = 1 / np.sqrt(L * C)

# Calcular el factor de calidad Q
Q = w_0 / R * L

#Ancho de banda
BW = w_0 / Q
f_bajo = w_0 - (BW/ 2)
f_alto = w_0 + (BW/ 2)

# Calcular el ángulo de fase phi (en radianes)
phi = np.arctan(1 / Q)

# La funcion np.longspace arma una funcion logatirmica con frecuencia espaciadas. 
frecuencias = np.logspace(-1, 3, 1000)  # Frecuencias de 0.1 a 1000 radianes/segundo

#Esta funcion lo que calcula es la respuestaa del sistema a distintas frecuencias.
w, mag, fase = signal.bode(sistema, frecuencias)           

#Asintota de altas frecuencia
asintota_altafreq = -20 * np.log10(frecuencias) + 20 * np.log10(R / L)

fase_asintota_alta = 90   # Fase en alta frecuencia

# Asintota bajas frecuencia (con pendiente +20 dB/década)
asintota_bajafreq = 20 * np.log10(frecuencias) 
fase_asintota_baja = -90  # Fase en baja frecuencia

#%%  Función de Transferencia Filtro 2

numerador2 = [1, 0, 0]
denominador2 = [1, 1/(R * C), 1 / (L * C)]

# Crear el sistema en el dominio de Laplace 
sistema2 = signal.TransferFunction(numerador2, denominador2)

# Calcular los polos y ceros del sistema
zeros2, poles2, ganancia2 = signal.tf2zpk(numerador2, denominador2)

# Calcular la frecuencia de resonancia w0
w_02 = 1 / np.sqrt(L * C)

# Calcular el factor de calidad Q
Q2 = (1 / R) * np.sqrt(L / C)

# Ancho de banda 
BW2 = w_02 / Q2
f_bajo2 = w_02 - (BW2 / 2)
f_alto2 = w_02 + (BW2 / 2)

# Calcular el ángulo de fase phi (en radianes)
phi2 = np.arctan(1 / Q2)

# La funcion np.logspace arma una funcion logarítmica con frecuencias espaciadas. 
frecuencias2 = np.logspace(-2, 5, 1000)  # Frecuencias de 0.1 a 1000 radianes/segundo

#Esta funcion lo que calcula es la respuesta del sistema a distintas frecuencias.
w2, mag2, fase2 = signal.bode(sistema2, frecuencias2)             

# Asintota de baja frecuencia
asintota_baja_freq2 = 40 * np.log10(1 / (R * C) * frecuencias2)
fase_asintota_baja2 = 0  # Fase en baja frecuencia


#Asintota de alta frecuencia
asintota_alta_freq2 = 20 * np.log10(R / L) * np.ones_like(frecuencias2)
fase_asintota_alta2 = 180   # Fase en alta frecuencia

#%% Visualizacion de resultados. 
# Mostrar resultados
print("Frecuencia de resonancia del filtro 1 (w0):", w_0)
print("Factor de calidad (Q):", Q)
print("Ángulo de fase (phi) en radianes: del filtro 1", phi)

# Mostrar los polos y ceros
print("Polos:", poles)
print("Ceros:", zeros)

print("Frecuencia de resonancia del filtro 2 (w0):", w_02)
print("Factor de calidad (Q):", Q2)
print("Ángulo de fase (phi) en radianes: del filtro 2", phi2)

# Mostrar los polos y ceros
print("Polos:", poles2)
print("Ceros:", zeros2)

#%% Graficar los polos y ceros en el plano de Laplace

#Filtro 1
plt.figure(1)
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Ceros', color='b')  # Ceros en azul
plt.scatter(np.real(poles), np.imag(poles), marker='x', label='Polos', color='r')  # Polos en rojo
plt.axhline(0, color='black',linewidth=1) #Eje imaginario
plt.axvline(0, color='black',linewidth=1) #Eje real
plt.title('Plano de Laplace: Polos y Ceros')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

#Filtro 2
plt.figure(2)
plt.scatter(np.real(zeros2), np.imag(zeros2), marker='o', label='Ceros', color='b')  # Ceros en azul
plt.scatter(np.real(poles2), np.imag(poles2), marker='x', label='Polos', color='r')  # Polos en rojo
plt.axhline(0, color='black',linewidth=1) #Eje imaginario
plt.axvline(0, color='black',linewidth=1) #Eje real
plt.title('Plano de Laplace: Polos y Ceros')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()


#%% Graficar módulo (en dB) y fase (en grados)
fig, (ax1, ax2) = plt.subplots(2, 1)

# Gráfico del módulo (en dB)
ax1.semilogx(w, mag, label="Módulo |H(jω)| (dB)")
ax1.semilogx(frecuencias, asintota_bajafreq, '--', label='Asíntota Baja Frecuencia')
ax1.semilogx(frecuencias, asintota_altafreq, '--', label='Asíntota Alta Frecuencia')
ax1.axvline(f_bajo, color='r', linestyle='--', label=f'Frecuencia Baja BW = {f_bajo:.2f} rad/s')
ax1.axvline(f_alto, color='r', linestyle='--', label=f'Frecuencia Alta BW = {f_alto:.2f} rad/s')
ax1.set_xlabel("Frecuencia [rad/s]")
ax1.set_ylabel("Amplitud [dB]")
ax1.set_xlim(1e-1, 1e3)  # Establecer el rango del eje x
ax1.set_ylim(-50, 10)  # Establecer el rango del eje y
ax1.grid(True)
ax1.legend()

# Gráfico de la fase (en grados)
ax2.semilogx(w, fase, label="Fase ∠H(jω) (°)")
ax2.axhline(fase_asintota_baja, color='g', linestyle='--', label=f'Asíntota Baja Frecuencia (fase={fase_asintota_baja}°)')
ax2.axhline(fase_asintota_alta, color='b', linestyle='--', label=f'Asíntota Alta Frecuencia (fase={fase_asintota_alta}°)')
ax2.set_xlabel("Frecuencia [rad/s]")
ax2.set_ylabel("Fase [grados]")
ax2.grid(True)

# Gráfico 2: Módulo y fase para el segundo conjunto de datos
fig, (ax1, ax2) = plt.subplots(2, 1)

# Gráfico del módulo (en dB)
ax1.semilogx(w2, mag2)
ax1.semilogx(frecuencias2, asintota_baja_freq2, '--', label='Asíntota Baja Frecuencia')
ax1.semilogx(frecuencias2, asintota_alta_freq2, '--', label='Asíntota Alta Frecuencia')
ax1.axvline(f_bajo2, color='r', linestyle='--', label=f'Frecuencia Baja BW = {f_bajo:.2f} rad/s')
ax1.axvline(f_alto2, color='r', linestyle='--', label=f'Frecuencia Alta BW = {f_alto:.2f} rad/s')
ax1.set_title("Respuesta en Frecuencia (Módulo en dB)")
ax1.set_xlabel("Frecuencia [rad/s]")
ax1.set_ylabel("Amplitud [dB]")
ax1.set_xlim(1e-1, 1e3)  # Establecer el rango del eje x
ax1.set_ylim(-50, 10)  # Establecer el rango del eje y
ax1.grid(True)
ax1.legend()

# Gráfico de la fase (en grados)
ax2.semilogx(w2, fase2,label="Fase ∠H(jω) (°)")
ax2.axhline(fase_asintota_baja2, color='g', linestyle='--', label=f'Asíntota Baja Frecuencia (fase={fase_asintota_baja}°)')
ax2.axhline(fase_asintota_alta2, color='b', linestyle='--', label=f'Asíntota Alta Frecuencia (fase={fase_asintota_alta}°)')
ax2.set_title("Respuesta en Frecuencia (Fase en grados)")
ax2.set_xlabel("Frecuencia [rad/s]")
ax2.set_ylabel("Fase [grados]")
ax2.grid(True)

# Mostrar las gráficas
plt.tight_layout()
plt.show()