# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:04:23 2026

@author: l
"""
#%% IMPORTAR LIBRERÍAS 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import stft, welch, find_peaks
from scipy.interpolate import interp1d
from scipy import signal
from pytc2.sistemas_lineales import plot_plantilla 
from scipy import stats

#%%
import importlib
import analizar_senales
importlib.reload(analizar_senales)
from pytc2.sistemas_lineales import plot_plantilla 
from analizar_senales import cargar_mediciones, graficar_tres_mediciones, graficar_superposicion, validar_muestreo, diccionario_interp, fs_ant_desp, analizar_voltaje, analizar_espectrograma_real, disenar_filtro, aplicar_filtro, aplicar_FIIR_ventana, graficar_filtrada, calcular_desfase_puro

#%%
ruta_eth= r"C:/Users/l/OneDrive - Universidad de San Martin/Documentos/FACU/APS/proyectos_python/TPFINAL/Nuevo enfoque--Pavía/Datos/ZETANCsSPIONetohvoltage.txt"
ruta_chl= r"C:/Users/l/OneDrive - Universidad de San Martin/Documentos/FACU/APS/proyectos_python/TPFINAL/Nuevo enfoque--Pavía/Datos/ZETANCsSPION_chl_voltage.txt"

#%%
mediciones_etanol = cargar_mediciones(ruta_eth)
mediciones_cloroformo = cargar_mediciones(ruta_chl)

# %%

graficar_tres_mediciones(mediciones_etanol, "Muestra Etanol")
graficar_tres_mediciones(mediciones_cloroformo, "Muestra Cloroformo")

# %%
graficar_superposicion(mediciones_etanol, "Etanol", zoom_lims=[1.15, 1.30, 0.03, 0.035])
graficar_superposicion(mediciones_cloroformo, "Cloroformo", zoom_lims=[1.15, 1.30, 0.0035, 0.007])

#%% 
validar_muestreo(mediciones_etanol['Promedio'], 'Ethanol')
#%%
validar_muestreo(mediciones_cloroformo['Promedio'],'Cloroformo')

#%% 
etanol_inter = diccionario_interp(mediciones_etanol, "Muestra Etanol")
validar_muestreo(etanol_inter['Promedio'],"Muestra Etanol")
#%%
cloroformo_inter = diccionario_interp(mediciones_cloroformo, "Muestra Cloroformo")
validar_muestreo(cloroformo_inter['Promedio'],"Muestra Cloroformo")

#%%
graficar_tres_mediciones(etanol_inter, "Muestra Etanol")
graficar_tres_mediciones(cloroformo_inter, "Muestra Cloroformo")
graficar_superposicion(etanol_inter, "Etanol", zoom_lims=[1.15, 1.30, 0.03, 0.035])
graficar_superposicion(cloroformo_inter, "Cloroformo", zoom_lims=[1.15, 1.30, 0.0035, 0.007])

#%%
analizar_voltaje(etanol_inter['Promedio'], "Muestra Etanol")
analizar_voltaje(cloroformo_inter['Promedio'], "Muestra Cloroformo")
#%%
fs_eth= fs_ant_desp(mediciones_etanol['Promedio'], etanol_inter['Promedio'], "Muestra Etanol")
fs_chl= fs_ant_desp(mediciones_cloroformo['Promedio'], cloroformo_inter['Promedio'], "Muestra Cloroformo")

#%% 
analizar_espectrograma_real(etanol_inter['Promedio'], fs_eth, "Muestra Etanol")
analizar_espectrograma_real(etanol_inter['Promedio'], fs_chl, "Muestra Cloroformo")

#%% 
sos_filtro_eth = disenar_filtro(fs_eth, lowcut=1.0, highcut=70.0)
sos_filtro_chl = disenar_filtro(fs_chl, lowcut=1.0, highcut=70.0)
#%%
etanol_df_filtrado = aplicar_filtro(etanol_inter['Promedio'], sos_filtro_eth, "Muestra Etanol", fs_eth)
cloroformo_df_filtrado = aplicar_filtro(cloroformo_inter['Promedio'], sos_filtro_chl, "Muestra Cloroformo", fs_chl)
#%%
graficar_filtrada(etanol_inter['Promedio'], etanol_df_filtrado, "Muestra Etanol")
graficar_filtrada(cloroformo_inter['Promedio'], cloroformo_df_filtrado, "Muestra Cloroformo")

#%% 
valor_desfase = calcular_desfase_puro(cloroformo_inter['Promedio'], fs_chl)
valor_desfase = calcular_desfase_puro(etanol_inter['Promedio'], fs_eth)
#%%
etanol_suave = aplicar_FIIR_ventana(etanol_inter['Promedio'],11, fs_eth)
cloroformo_suave = aplicar_FIIR_ventana(cloroformo_inter['Promedio'], 11, fs_chl)
#%% 
graficar_filtrada(etanol_inter['Promedio'], etanol_suave, "Muestra Etanol")
graficar_filtrada(cloroformo_inter['Promedio'], cloroformo_suave, "Muestra Cloroformo")
#%% 