# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:24:47 2026

@author: l
"""


#%% IMPORTAR LIBRERÍAS

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Leer y gráficas datos 

voltagechl= r"C:/Users/l/OneDrive - Universidad de San Martin/Documentos/FACU/APS/proyectos_python/TPFINAL/Nuevo enfoque--Pavía/Datos/ZETANCsSPION_chl_voltage.txt"
df_chl = pd.read_csv(voltagechl, sep='\t')
print(df_chl)

#%% 
#** Se observa en el explorador de variables, que el sistema trabaja recolentando la informacion en tres etapas: Time, Time.1 y Time2. Finalmente hace una columna final, como promedio de los datos anteriores. 
# Bloque 1: Time + Voltage y Time + Current (Medición 1).
# Bloque 2: Time.1 + Voltage.1 y Time.1 + Current.1 (Medición 2).
# Bloque 3: Time.2 + Voltage.2 y Time.2 + Current.2 (Medición 3).
# Bloque 4: Columnas (Avg) (El promedio que calcula el equipo). 

# Separamos las mediciones y las guardamos en diferentes dataframes
medicion1_chl = ['Time (s) - Voltage (NCs1 CHCl3 A40 + SPION esano)', 'Voltage (V) - Voltage (NCs1 CHCl3 A40 + SPION esano)', 'Current (mA) - Current (NCs1 CHCl3 A40 + SPION esano)']

medicion2_chl = ['Time (s) - Voltage (NCs1 CHCl3 A40 + SPION esano).1', 'Voltage (V) - Voltage (NCs1 CHCl3 A40 + SPION esano).1', 'Current (mA) - Current (NCs1 CHCl3 A40 + SPION esano).1']

medicion3_chl = ['Time (s) - Voltage (NCs1 CHCl3 A40 + SPION esano).2', 'Voltage (V) - Voltage (NCs1 CHCl3 A40 + SPION esano).2', 'Current (mA) - Current (NCs1 CHCl3 A40 + SPION esano).2']

# 2. Creamos los 4 DataFrames
df1_chl = df_chl[medicion1_chl].dropna()
df2_chl = df_chl[medicion2_chl].dropna()
df3_chl = df_chl[medicion3_chl].dropna()

# Para el Avg (promedio), buscamos las columnas que dicen (Avg)
cols_avg_chl = [col for col in df_chl.columns if '(Avg)' in col]
df_avg_chl = df_chl[cols_avg_chl].dropna()


#%% Graficos de señales crudas
#VoltajevsCorriente en fucnion del tiempo

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df1_chl.iloc[:, 0], df1_chl.iloc[:, 1], color='tab:blue', label='Voltaje')
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Voltaje (V)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')


ax2 = ax1.twinx()
ax2.plot(df1_chl.iloc[:, 0], df1_chl.iloc[:, 2], color='tab:red', label='Corriente', alpha=0.7)
ax2.set_ylabel('Corriente (mA)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Señales de Voltaje y Corriente en función del tiempo')
fig1.tight_layout()
plt.show()

#%% 
plt.figure(figsize=(6, 6))
plt.plot(df1_chl.iloc[:, 1], df1_chl.iloc[:, 2], color='purple', alpha=0.5)
plt.title("Corriente vs. Voltaje")
plt.xlabel("Voltaje (V)")
plt.ylabel("Corriente (mA)")
plt.grid(True)
plt.show()

#%% 

def analizar_toda_la_señal(df):
    tiempo = df_chl.iloc[:, 0].values
    i = df_chl.iloc[:, 2].values
    v = df_chl.iloc[:, 1].values
    
    # 1. Detectar flancos (igual que antes)
    flancos = np.where((v[1:] > v[:-1] + 0.5))[0]
    
    picos = []
    estacionarios = []
    
    #Como mi dt no es constante, necesito determinar ventanas para detectar los picos!
    ventana_pico = 0.05  
    ventana_ss_inicio = 0.1 
    ventana_ss_fin = 0.3    
    
    for idx in flancos:
        t_inicio = tiempo[idx]
        idx_pico_fin = np.searchsorted(tiempo, t_inicio + ventana_pico)
        idx_ss_inicio = np.searchsorted(tiempo, t_inicio + ventana_ss_inicio)
        idx_ss_fin = np.searchsorted(tiempo, t_inicio + ventana_ss_fin)
        if idx_ss_fin < len(i):
            picos.append(np.max(i[idx:idx_pico_fin]))
            estacionarios.append(np.mean(i[idx_ss_inicio:idx_ss_fin]))
            
    #Etsadísticas de interés
    i_peak_promedio = np.mean(picos)
    i_ss_promedio = np.mean(estacionarios)
    
    # Desvío estándar 
    std_ruido_global = np.std(i)
    
    # SNR y Factor de forma
    snr = i_peak_promedio / std_ruido_global
    factor_forma = i_ss_promedio / i_peak_promedio
    
    return i_peak_promedio, i_ss_promedio, snr, factor_forma, std_ruido_global

# Y lo llamás así:
i_peak, i_ss, snr, factor, ruido = analizar_toda_la_señal(df_chl)
print(f"Pico Promedio: {i_peak:.4f}, SNR: {snr:.2f}, Factor: {factor:.4f}")
