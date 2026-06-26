# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:05:35 2026

@author: l
"""
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
from scipy.signal import stft, welch, find_peaks
from scipy import signal
from pytc2.sistemas_lineales import plot_plantilla 

#%% FUNCIONES

def cargar_mediciones(ruta):
    df_raw = pd.read_csv(ruta, sep='\t')
    diccionario_mediciones = {}
    #Cargar datos de las mediciones en un diccionario(más ordenado)
    for i, nombre in enumerate(['Medicion_1', 'Medicion_2', 'Medicion_3', 'Promedio']):
        idx = i * 4
        bloque = df_raw.iloc[:, idx : idx+4]
        df_limpio = pd.DataFrame({
            'Time': bloque.iloc[:, 0],
            'Voltage': bloque.iloc[:, 1],
            'Current': bloque.iloc[:, 3]
        }).dropna()
        
        diccionario_mediciones[nombre] = df_limpio
        
    return diccionario_mediciones

def graficar_tres_mediciones(diccionario_muestras, nombre_grafico):
    num_mediciones = len(diccionario_muestras)
    fig, axs = plt.subplots(num_mediciones, 1, figsize=(10, 3 * num_mediciones), sharex=True)
    if num_mediciones == 1:
        axs = [axs]
        
    for i, (nombre_medicion, df) in enumerate(diccionario_muestras.items()):
        ax_v = axs[i]
        
        # Voltaje
        ax_v.plot(df['Time'], df['Voltage'], color='tab:blue', label='Voltaje')
        ax_v.set_ylabel('Voltaje (V)', color='tab:blue')
        ax_v.set_title(f"{nombre_grafico} - {nombre_medicion}")
        ax_v.grid(True, alpha=0.3)
        
        # Corriente (eje secundario)
        ax_c = ax_v.twinx()
        ax_c.plot(df['Time'], df['Current'], color='tab:red', alpha=0.7, label='Corriente')
        ax_c.set_ylabel('Corriente (mA)', color='tab:red')
        
    axs[-1].set_xlabel('Tiempo (s)')
    fig.tight_layout()
    plt.show()
# %%

def graficar_superposicion(diccionario, nombre_muestra, zoom_lims=None):
    mediciones_list = [df['Current'].values for nombre, df in diccionario.items() if nombre != 'Promedio']
    promedio_vals = diccionario['Promedio']['Current'].values
    std_dev = np.std(mediciones_list, axis=0)
    std_medio = np.mean(std_dev)
    pico_max = np.max(np.abs(promedio_vals))
    error_porcentual = (std_medio / pico_max) * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for nombre, df in diccionario.items():
        if nombre == 'Promedio':
            estilo = {'color': 'black', 'linestyle': '--', 'linewidth': 2, 'alpha': 1.0}
        else:
            estilo = {'linewidth': 1, 'alpha': 0.6}
            
        ax1.plot(df['Time'], df['Current'], label=nombre, **estilo)
        ax2.plot(df['Time'], df['Current'], label=nombre, **estilo)
        
    texto_metrica = f"Consistencia: Desvío Estándar Medio = {std_medio:.4f} mA\n" \
                    f"Error Relativo = {error_porcentual:.2f} %"
    ax1.text(0.02, 0.95, texto_metrica, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    # Vista General
    ax1.set_title(f'Superposición de Corrientes: {nombre_muestra}')
    ax1.set_ylabel('Corriente (mA)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zoom
    ax2.set_title('Zoom: Transitorio de carga')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Corriente (mA)')
    
    if zoom_lims:
        ax2.set_xlim(zoom_lims[0], zoom_lims[1])
        ax2.set_ylim(zoom_lims[2], zoom_lims[3])
        
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
#### Se ve claramente que la columna promedio es consistente con el las tres mediciones, denota ademas un desvio estandar de entre 0.7-3% (para ethanol y cloroformo) de desvio estandar por lo que muchos de los siguientes analisis SÓLO se haran sobre el promedio (no todo)

def graficar_filtrada(df_original, df_filtrado, nombre_muestra):
    plt.figure(figsize=(10, 5))
    plt.plot(df_original['Time'], df_original['Current'], color='tab:blue', alpha=0.5, label='Corriente sin filtrar')
    plt.plot(df_filtrado['Time'], df_filtrado['Current'], color='tab:red', linewidth=1.5, label='Corriente filtrada')
    plt.title(f"Comparación Antes vs. Después del Filtrado: {nombre_muestra}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Corriente [mA]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
#%%
    
def validar_muestreo(df, nombre_muestra):
    diffs = np.diff(df['Time'])
    fluctuacion = np.std(diffs) / np.mean(diffs)
    v = df['Voltage'].values
    t = df['Time'].values
    cambios_signo = np.where(np.diff(np.sign(v)))[0]
    periodos = np.diff(t[cambios_signo])
    diferencia_periodos = np.diff(periodos)
    indice_cambio = np.argmax(np.abs(diferencia_periodos)) + 1
    tiempo_corte = t[cambios_signo[indice_cambio]]
    
    df_rapido = df[df['Time'] < tiempo_corte]
    df_lento = df[df['Time'] >= tiempo_corte]

    # Calculamos los dt por separado
    diffs_rapido = np.diff(df_rapido['Time'])
    diffs_lento = np.diff(df_lento['Time'])
    
    print(f"--- Análisis de Regímenes ---")
    print(f"Desviación estándar dt rápido: {np.std(diffs_rapido):.8f}")
    print(f"Desviación estándar dt lento: {np.std(diffs_lento):.8f}")
    
    print(f"--- Diagnóstico de Muestreo ---")
    print(f"Mean dt: {np.mean(diffs):.6f} s")
    print(f"Fluctuacion temporal: {fluctuacion:.4f}")
    
    plt.figure(1, figsize=(8, 4))
    plt.hist(diffs, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(diffs), color='red', linestyle='--', label='Promedio')
    plt.title(f"Distribución del intervalo de muestreo (dt): {nombre_muestra}")
    plt.xlabel("Intervalo de tiempo (s)")
    plt.ylabel("Frecuencia (cantidad de muestras)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    plt.figure(2, figsize=(8, 4))
    plt.plot(diffs, color='purple', alpha=0.7)
    plt.axhline(np.mean(diffs), color='red', linestyle='--', label='Promedio')
    plt.title(f"Variabilidad del dt: {nombre_muestra}")
    plt.ylabel("Diferencia de tiempo (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    if fluctuacion > 0.01: 
        print("ALERTA: El dt no es constante. Se requiere resampleo.")
    else:
        print("Muestreo estable: dt constante aceptable.")
    return 
    
#%%
def diccionario_interp(diccionario, nombre):
    diccionario_interp= {}
    
    for nombre, df in diccionario.items():
        diffs = np.diff(df['Time'])
        dt_moda = stats.mode(diffs[diffs > 1e-6], keepdims=True).mode[0]
        fs_ideal = 1 / dt_moda
        print(f"--- Interpolando {nombre} ---")
        print(f"fs detectada automáticamente: {fs_ideal:.2f} Hz")
        t_inicio = df['Time'].min()
        t_fin = df['Time'].max()
        t_nuevo = np.arange(t_inicio, t_fin, 1/fs_ideal) #Creamos un dt nuevo
        
        # Creamos las funciones de interpolación
        f_volt = interp1d(df['Time'], df['Voltage'], kind='linear', fill_value="extrapolate")
        f_curr = interp1d(df['Time'], df['Current'], kind='linear', fill_value="extrapolate")
        
        df_interpolado = pd.DataFrame({
            'Time': t_nuevo,
            'Voltage': f_volt(t_nuevo),
            'Current': f_curr(t_nuevo)
        })
        
        diccionario_interp[nombre] = df_interpolado
        
    return diccionario_interp

# Mi señal se encuentra en 1Hz y 19Hz de fundamental, una cuadrada necesita  al menos 5 armonicos para mantener su naturaleza por lo que la fs de 1000Hz encontrada es coherente y no corro riesgo de aliasing, cuando los grafico coinciden claramente en frecuencia los picos solo disminuyo la magnitud, y reduzco 1/3 de muestras esto puede ser porque habia un oversampling, se reducio drasticamente el "ruido" de alta frecuencia 
#%% 
def analizar_voltaje(df,nombre_muestra ):
    v = df['Voltage'].values
    t = df['Time'].values
    cambios_signo = np.where(np.diff(np.sign(v)))[0]
    periodos = np.diff(t[cambios_signo])
    # Cambio de regimen 
    indice_cambio = np.argmax(np.abs(np.diff(periodos))) + 1
    t_corte = t[cambios_signo[indice_cambio]]
    
    df_rapido = df[df['Time'] < t_corte]
    df_lento = df[df['Time'] >= t_corte]
    
    def calcular_fisica_voltaje(df_seg):
        v = df_seg['Voltage'].values
        t = df_seg['Time'].values
        cambios = np.where(np.diff(np.sign(v)))[0]
        if len(cambios) > 1:
            freq = (len(cambios) / 2) / (t[-1] - t[0])
        else:
            freq = 0
        amp = (v.max() - v.min()) / 2
        return amp, freq
    
    amp_r, freq_r = calcular_fisica_voltaje(df_rapido)
    amp_l, freq_l = calcular_fisica_voltaje(df_lento)
    
    print(f"--- Análisis Físico: {nombre_muestra} ---")
    print(f"Régimen Rápido -> Frecuencia: {freq_r:.2f} Hz | Amplitud: {amp_r:.3f} V")
    print(f"Régimen Lento  -> Frecuencia: {freq_l:.2f} Hz | Amplitud: {amp_l:.3f} V\n")

#%%

def fs_ant_desp(df, df_interp, nombre):
        # FFT de la original
        dt = np.mean(np.diff(df['Time']))
        fs = 1 / dt
        
        n = len(df)
        fft = np.abs(np.fft.fft(df['Current'].values))
        freqs= np.fft.fftfreq(n, d=1/fs)
        picos_df, _ = find_peaks(fft[:n//2], height=np.max(fft[:n//2])*0.1)
        
        ##FFT de la inteprolada
        dt_interp = np.mean(np.diff(df_interp['Time']))
        fs_ideal = 1 / dt_interp
        
        n_interp = len(df_interp)
        fft_interp = np.abs(np.fft.fft(df_interp['Current'].values))
        freqs_interp = np.fft.fftfreq(n_interp, d=1/fs_ideal)
        picos_interp, _ = find_peaks(fft_interp[:n_interp//2], height=np.max(fft_interp[:n_interp//2])*0.1)
        
        plt.figure(figsize=(10, 5))
        plt.plot(freqs[:n//2], fft[:n//2], label=f'Cruda (fs={fs:.0f} Hz)', alpha=0.6)
        plt.scatter(freqs[picos_df], fft[picos_df], color='blue', s=30)
        plt.plot(freqs_interp[:n_interp//2], fft_interp[:n_interp//2], label=f'Interpolada (fs={fs_ideal:.0f} Hz)', alpha=0.9)
        plt.scatter(freqs_interp[picos_interp], fft_interp[picos_interp], color='red', s=30)
        
        plt.title(f"Validación Espectral: {nombre}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")
        plt.xlim(0, 150) # Ajustá según tu interés
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return fs_ideal
    
        
def analizar_espectrograma_real(df, fs, nombre):
    i = df.iloc[:, 2].values
    v = df.iloc[:, 1].values
    f, t_stft, Zxx = stft(i, fs=fs, nperseg=256)
    f2, t_stft_v, Zxx_v = stft(v, fs=fs, nperseg=256)
    
    # Graficar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    pcm = ax1.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    ax1.set_title("Espectrograma de corriente: {nombre}")
    ax1.set_ylabel("Frecuencia [Hz]")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylim(0, 150) # Ajustá esto según veas tus picos en la FFT
    fig.colorbar(pcm, ax=ax1, label="Magnitud")
    
    pcm2 = ax2.pcolormesh(t_stft_v, f2, np.abs(Zxx_v), shading='gouraud', cmap='viridis')
    ax2.set_title("Espectrograma de voltaje")
    ax2.set_ylabel("Frecuencia [Hz]")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylim(0, 150) 
    fig.colorbar(pcm2, ax=ax2, label="Magnitud")
    
    plt.tight_layout()
    plt.show()
    
#%% 

def disenar_filtro(fs, lowcut=1.0, highcut=70.0):
    fpass = np.array([lowcut, highcut])
    fstop = np.array([0.5, 90.0]) 
    orden = 4
    sos = signal.butter(orden, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    w, h = signal.sosfreqz(sos, worN=8000, fs=fs)
    plt.figure(figsize=(8, 5))
    plt.plot(w, 20 * np.log10(np.abs(h) + 1e-15), color='blue', linewidth=2, label=f'Butterworth Orden {orden}')
    plt.title(f'Respuesta en Frecuencia: Orden {orden}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [dB]')
    plt.xlim(0, 150)
    plt.ylim(-60, 5)
    plt.grid(True, which='both')
    plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=1, fstop=fstop, attenuation=40, fs=fs)
    plt.legend()
    plt.show()
    
    return sos

def aplicar_filtro(df, sos, nombre_muestra, fs):
    df_filtrado = df.copy()
    i_filtrada = signal.sosfiltfilt(sos, df['Current'].values)
    df_filtrado['Current'] = i_filtrada
    
    # Cálculos para FFT
    n = len(df)
    freqs = np.fft.fftfreq(n, d=1/fs)
    mag_original = np.abs(np.fft.fft(df['Current'].values))[:n//2]
    mag_filtrada = np.abs(np.fft.fft(i_filtrada))[:n//2]
    freqs_pos = freqs[:n//2]
    
    # Graficar Comparación Espectral
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_pos, mag_original, color='gray', alpha=0.5, label='Original')
    plt.plot(freqs_pos, mag_filtrada, color='green', label='Filtrada (Butterworth)')
    plt.title(f'Validación de Filtrado: {nombre_muestra}')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.xlim(0, 150)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return df_filtrado
# Filtro IIR no me sirve! Introduce deformaciones en mi señal, principalmente en los flancos de la señal cuadrada y en el regimen lento de una sola onda. Por lo tanto, voy a calcular el desfase entre señales y aplicar un filtro FIIR que es lineal y de fase 0 con metodo de ventanas
def calcular_desfase_puro(df, fs):
    v = df['Voltage'].values
    i = df['Current'].values
    n = len(v)
    
    fft_v = np.fft.fft(v)
    fft_i = np.fft.fft(i)
    freqs = np.fft.fftfreq(n, d=1/fs)
    
    # Filtramos la fundamental (cercana a 19Hz)
    mask = (freqs > 10.0) & (freqs < 100.0)
    idx_f0 = np.argmax(np.abs(fft_v[mask]))
    
    # Extraemos ángulos
    fase_v = np.angle(fft_v[mask][idx_f0])
    fase_i = np.angle(fft_i[mask][idx_f0])
    
    # Calcular grados
    desfase = np.degrees(fase_i - fase_v)
    if desfase < -180: desfase += 360
    if desfase > 180: desfase -= 360
    
    print(f"El desfase real es: {desfase:.2f} grados")

#%% 
#Diseño filtro FIIR por metodo de ventanas, voy a ir por la rectangular. El ruido se encuentro en la parte "estacionaria" de la señal cuando sube o baja el voltaje. Mis nanocapsulas se representan como mini capacitores (por tener doble capa) entonces voy a analizar este comportamiento mas adelante. Para ello debo limpiar el ruido en esta seccion, por esoy voy a usar una ventana cuadrada ya que esto me ayudara a no solo no deformar la fase tampoco introduce retardo de grupo y no me afecta el comportamiento de los flancos. Este ruido es de alta frecuencia, electroestatica seguramente. 
def aplicar_FIIR_ventana(df,ventana,fs):
    df_filtrado = df.copy()
    rectangular = np.ones(ventana) / ventana
    df_filtrado['Current'] = np.convolve(df['Current'].values, rectangular, mode='same')
    b = rectangular 
    a = [1] 
    
    w, h = signal.freqz(b, a, worN=8000, fs=fs)
    fpass = 140.0    # Por mis armonicos si mi fundamental es 18-19Hz quiero dejar al menos 4 armonicos
    fstop = 200.0  
    
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h) + 1e-15), color='red', label='Respuesta FIR')
    plot_plantilla(filter_type="lowpass", fpass=fpass, fstop=fstop, ripple=1, attenuation=20, fs=fs)
    plt.title('Respuesta en Frecuencia con Plantilla')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [dB]')
    plt.xlim(0, 200)
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_filtrado

#Ventana 7 el lobulo principal se ajusta a la plantilla. Con 11 se hace mas angosto y entran mas lobulos laterales en la plantilla. (Voy a probar otros ordenes y ver su comportamiento) Ninguno afecta los flancos por ahora
#%% 

#Fantan dellates arreglar ---> quiero analizar la propiedad capacitiva de mis muestras ¿Como? Modelizando como capacitores, analizo los flancos y busco el tau para ver cual tiene mas capacitancia y determinar el mejor solvente por capacidades electricas. Ademas, el "ruido" que genera cada uno en el estacionario tambien me da informaicon de que solvente es mejor. 
def analizar_toda_la_señal(df):
    tiempo = df.iloc[:, 0].values
    i = df.iloc[:, 2].values
    v = df.iloc[:, 1].values
    
    #Quiero analizar el efecto capacitivo de la corriente en cada subida de volttaje, por lo que identifico dos flacos: subida y bajada
    flancos_positivos = np.where((v[1:] > 140) & (v[:-1] < 140))[0]
    flancos_negativos = np.where((v[1:] < -140) & (v[:-1] > -140))[0]
    flancos = np.sort(np.concatenate([flancos_positivos, flancos_negativos]))
    
    picos = []
    estacionarios = []
    
# --> Esto fue antes de interpolar, tengo que cambiar el codigo     #Como mi dt no es constante, necesito determinar ventanas para detectar los picos ya que no puedo confiar en mi dt! Etsablezco para probar: ajustar segun resultados
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