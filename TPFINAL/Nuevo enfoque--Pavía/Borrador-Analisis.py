# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:49:17 2026

@author: l
"""

#%% IMPORTAR LIBRERÍAS

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import stft, welch
from scipy.interpolate import interp1d
from scipy import signal
from pytc2.sistemas_lineales import plot_plantilla 
from scipy.signal import welch
#%% Leer y gráficas datos 

voltageeth= r"C:/Users/l/OneDrive - Universidad de San Martin/Documentos/FACU/APS/proyectos_python/TPFINAL/Nuevo enfoque--Pavía/Datos/ZETANCsSPIONetohvoltage.txt"
df = pd.read_csv(voltageeth, sep='\t')
print(df)

#%% Separo las mediciones en diferentes dataframes
#** Se observa en el explorador de variables, que el sistema trabaja recolentando la informacion en tres etapas: Time, Time.1 y Time2. Finalmente hace una columna final, como promedio de los datos anteriores. 
# Bloque 1: Time + Voltage y Time + Current (Medición 1).
# Bloque 2: Time.1 + Voltage.1 y Time.1 + Current.1 (Medición 2).
# Bloque 3: Time.2 + Voltage.2 y Time.2 + Current.2 (Medición 3).
# Bloque 4: Columnas (Avg) (El promedio que calcula el equipo). 

# Separamos las mediciones y las guardamos en diferentes dataframes
medicion1 = ['Time (s) - Voltage (NCs1 ETOH A40 SPION ESANO )', 'Voltage (V) - Voltage (NCs1 ETOH A40 SPION ESANO )', 'Current (mA) - Current (NCs1 ETOH A40 SPION ESANO )']

medicion2 = ['Time (s) - Voltage (NCs1 ETOH A40 SPION ESANO ).1', 'Voltage (V) - Voltage (NCs1 ETOH A40 SPION ESANO ).1', 'Current (mA) - Current (NCs1 ETOH A40 SPION ESANO ).1']

medicion3 = ['Time (s) - Voltage (NCs1 ETOH A40 SPION ESANO ).2', 'Voltage (V) - Voltage (NCs1 ETOH A40 SPION ESANO ).2', 'Current (mA) - Current (NCs1 ETOH A40 SPION ESANO ).2']

# 2. Creamos los 4 DataFrames
df1 = df[medicion1].dropna()
df2 = df[medicion2].dropna()
df3 = df[medicion3].dropna()

# Para el Avg (promedio), buscamos las columnas que dicen (Avg)
cols_avg = [col for col in df.columns if '(Avg)' in col]
df_avg = df[cols_avg].dropna()


#%% Graficos de señales crudas
#VoltajevsCorriente en fucnion del tiempo

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df1.iloc[:, 0], df1.iloc[:, 1], color='tab:blue', label='Voltaje')
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Voltaje (V)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')


ax2 = ax1.twinx()
ax2.plot(df1.iloc[:, 0], df1.iloc[:, 2], color='tab:red', label='Corriente', alpha=0.7)
ax2.set_ylabel('Corriente (mA)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Señales de Voltaje y Corriente en función del tiempo')
fig1.tight_layout()
plt.show()

#%% Grafico todas las mediciones con su corriente y voltaje en una sola figura 
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

datos = [(df1, 'Medición 1'), (df2, 'Medición 2'), (df3, 'Medición 3')]

for i, (dfs, titulo) in enumerate(datos):
    ax_v = axs[i]
    ax_v.plot(dfs.iloc[:, 0], dfs.iloc[:, 1], color='tab:blue', label='Voltaje') 
    ax_v.set_ylabel('Voltaje (V)', color='tab:blue')
    ax_v.set_title(titulo)
    ax_v.grid(True, alpha=0.3)
    
    ax_c = ax_v.twinx()
    # CAMBIA df POR dfs AQUÍ ABAJO:
    ax_c.plot(dfs.iloc[:, 0], dfs.iloc[:, 2], color='tab:red', alpha=0.7, label='Corriente')
    ax_c.set_ylabel('Corriente (mA)', color='tab:red')
    
    if i == 0:
        ax_v.legend(loc='upper left')
        ax_c.legend(loc='upper right')

axs[2].set_xlabel('Tiempo (s)')
fig.tight_layout()
plt.show()

#%% Grafico de las tres corrientes crudas 
# Hipotesis: Si las tres líneas se superponen, el sistema es estable. Si una línea se desplaza o tiene una amplitud diferente, es deriva temporal 
# La corriente tiene un pico en la subido del voltaje esto se puede deber a la naturaleza capacitiva de las capsular (doble capa)
# Creamos una figura con 2 filas y 1 columna
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

#Corrientes total
ax1.plot(df1.iloc[:, 0], df1.iloc[:, 2], label='Medición 1', alpha=0.6)
ax1.plot(df2.iloc[:, 0], df2.iloc[:, 2], label='Medición 2', alpha=0.6)
ax1.plot(df3.iloc[:, 0], df3.iloc[:, 2], label='Medición 3', alpha=0.6)
ax1.set_title('Superposición de Corrientes (Vista General)')
ax1.set_ylabel('Corriente (mA)')
ax1.legend()
ax1.grid(True, alpha=0.3)

#Zoom: transitorio
ax2.plot(df1.iloc[:, 0], df1.iloc[:, 2], label='Medición 1', alpha=0.6)
ax2.plot(df2.iloc[:, 0], df2.iloc[:, 2], label='Medición 2', alpha=0.6)
ax2.plot(df3.iloc[:, 0], df3.iloc[:, 2], label='Medición 3', alpha=0.6)
ax2.set_title('Zoom: Transitorio de carga en la subida')
ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Corriente (mA)')
ax2.set_xlim(1.15, 1.30)     
ax2.set_ylim(0.03, 0.035)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%% Como el software actual no permite acceder a los datos de configuaracion, se puede determinar, como analisis incicial la frecuencia de muestreo a partir de los datos del tiempo fs=1/dt
tiempo = df1.iloc[:, 0]
dt = tiempo.iloc[1] - tiempo.iloc[0]
fs = 1 / dt
print(f"La frecuencia de muestreo calculada es: {fs:.2f} Hz")

ancho_banda= fs/2
print(f"El ancho de banda del Ethanol es: {ancho_banda:.2f} Hz")

#%%Observo el espectro de frecuencia 
t = df1.iloc[:, 0] 
v = df1.iloc[:, 1]
i = df1.iloc[:, 2]

# Cálculo de la FFT
n = len(t)
f_s = 1 / (t[1] - t[0]) # Frecuencia de muestreo experimental
fft_v = np.fft.fft(v)
frecuencias = np.fft.fftfreq(n, d=1/f_s)
fft_i = np.fft.fft(i)

# Filtramos solo la mitad positiva y normalizamos
magnitud_v = np.abs(fft_v)[:n//2]
magnitud_i = np.abs(fft_i)[:n//2]
freq_pos = frecuencias[:n//2]

#%% Graficamos el espectro de corriente
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. Espectro de Corriente (ax1)
ax1.plot(freq_pos, magnitud_i, color='tab:red', label='Corriente')
ax1.axvline(x=50, color='k', linestyle='--', label='Ruido 50 Hz')
ax1.set_title("Espectro de Frecuencia: Corriente")
ax1.set_ylabel("Magnitud")
ax1.set_xlim(0, 500)
ax1.grid(True)
ax1.legend()

# 2. Espectro de Voltaje (ax2)
ax2.plot(freq_pos, magnitud_v, color='tab:blue', label='Voltaje')
ax2.axvline(x=50, color='k', linestyle='--', label='Ruido 50 Hz')
ax2.set_title("Espectro de Frecuencia: Voltaje")
ax2.set_xlabel("Frecuencia (Hz)")
ax2.set_ylabel("Magnitud")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

#%% Marco los picos con magnitud por encima de 20 para separarlo de los armonicos y ruido menor
umbral = 20
indices_picos = np.where(magnitud_i > umbral)[0]

plt.figure(figsize=(10, 5))
plt.plot(freq_pos, magnitud_i, color='tab:red', label='Corriente')
plt.scatter(freq_pos[indices_picos], magnitud_i[indices_picos], color='black', s=20, label='Picos > 20')
plt.axhline(y=umbral, color='gray', linestyle=':', label='Umbral de ruido (20)')
plt.title("Espectro de Corriente con umbral de detección")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.xlim(0, 500)
plt.legend()
plt.grid(True)
plt.show()

#%% Observamos claramente dos picos en la señal, esto se dbee a que como calculamos previamente, existen dos regimenes en la señal. La frecuencia de la señal cambia con el tiempo, por esto vamos a observar esto con un espectrograma
#%% ESPECTROGRAMA
def analizar_espectrograma_real(df, fs):
    i = df.iloc[:, 2].values
    v = df.iloc[:, 1].values
    f, t_stft, Zxx = stft(i, fs=fs, nperseg=256)
    f2, t_stft_v, Zxx_v = stft(v, fs=fs, nperseg=256)
    
    # Graficar
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    pcm = ax1.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    ax1.set_title("Espectrograma de corriente")
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

# Llamalo con tu señal completa:
analizar_espectrograma_real(df1, fs)
#%% Calcular PSD
freqs, psd = welch(df.iloc[:, 2], fs=1000) # Asumiendo fs = 1000Hz

plt.figure(figsize=(8, 4))
plt.semilogy(freqs, psd)
plt.title("Densidad Espectral de Potencia (PSD)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia (V^2/Hz)")
plt.grid(True)
plt.show()

#%% Se que mi señal opera en dos regimenes: "rapido" y "lento". En una señal cuadrada, cada vez que pasa por 0 es un periodo. Busco los cruces por 0
v = df1.iloc[:, 1].values
t = df1.iloc[:, 0].values

cambios_signo = np.where(np.diff(np.sign(v)))[0]
periodos = np.diff(t[cambios_signo]) #Esto es cuanto dura cada uno de mis ondas! 
#AAAhora, comparo, cuando cambia el periodo! Tambien cambia mi frecuancia (obviamente)
diferencia_periodos = np.diff(periodos)
indice_cambio = np.argmax(np.abs(diferencia_periodos)) + 1
tiempo_corte = t[cambios_signo[indice_cambio]]

print(f"El cambio de régimen ocurre automáticamente en: {tiempo_corte:.4f} s")

#%% Comprobamos si reconoce el cambio: 
    
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df1.iloc[:, 0], df1.iloc[:, 1], color='tab:blue', label='Voltaje')
plt.scatter(tiempo_corte, 0, color='black', marker='o', s=100)
ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Voltaje (V)', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')


ax2 = ax1.twinx()
ax2.plot(df1.iloc[:, 0], df1.iloc[:, 2], color='tab:red', label='Corriente', alpha=0.7)
ax2.set_ylabel('Corriente (mA)', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Señales de Voltaje y Corriente en función del tiempo')
fig1.tight_layout()
plt.show()


#%% Ahora, separamos en los dos regimenes: 
df_rapido = df1[df1.iloc[:, 0] < tiempo_corte]
df_lento = df1[df1.iloc[:, 0] >= tiempo_corte]
    
#%% Ahora determino los datos de mi señal de voltaje que es mi señal REFERENCIA, si el sistema es estable la corriente deberia asemejarse a esta señal ya que es la "ideal" la que inyecta el sistema
def analizar_voltaje(df):
    v = df.iloc[:, 1].values
    t = df.iloc[:, 0].values
    cambios = np.where(np.diff(np.sign(v)))[0]
    if len(cambios) > 1:
        tiempo_seg = t[cambios[-1]] - t[cambios[0]]
        frecuencia= (len(cambios) / 2) / tiempo_seg
    amp = (df.iloc[:, 1].max() - df.iloc[:, 1].min()) / 2
    std_v = df.iloc[:, 1].std()
    return amp, std_v, frecuencia

ampfreq_rapida, stdfreq_rapida, freq_rapida = analizar_voltaje(df_rapido)
ampfreq_lenta, stdfreq_lenta, freq_lenta = analizar_voltaje(df_lento)

print(f"Frecuencia Régimen 1: {freq_rapida:.2f} Hz")
print(f"Frecuencia Régimen 2: {freq_lenta:.2f} Hz")
print(f"Amplitud medida Regimen 1: {ampfreq_rapida:.2f} V")
print(f"Amplitud medida Regimen 2: {ampfreq_lenta:.2f} V")
print(f"Desvío estándar  Regimen 1: {stdfreq_rapida:.2f} V")
print(f"Desvío estándar Regimen 2: {stdfreq_lenta:.2f} V")

#%% Sospecho que la frecuencia de uestrei NO es estable grafico
diferencias_tiempo = np.diff(df1.iloc[:, 0].values)
plt.plot(diferencias_tiempo)
plt.title("Variabilidad del tiempo de muestreo (dt)")
plt.ylabel("diferencia de tiempo (s)")
plt.show()
#%% Observo los dos regimenes por separado 

#%% Regimen alto, puedo hacer fft porque tengo una sinusoidal con armonicos,en el lento por otor lado no. Es una señal escalonada con estados estacionarios la fft no funciona
t_rapido = df_rapido.iloc[:, 0] 
v_rapido= df_rapido.iloc[:, 1]
i_rapido = df_rapido.iloc[:, 2]

# Cálculo de la FFT

n_rapido = len(t_rapido)
f_s_rapido = 1 / (t_rapido[1] - t_rapido[0]) # Frecuencia de muestreo experimental
fft_v_rapido = np.fft.fft(v_rapido)
frecuencias_rapido = np.fft.fftfreq(n_rapido, d=1/f_s_rapido)
fft_i_rapido = np.fft.fft(i_rapido)

# Filtramos solo la mitad positiva y normalizamos
magnitud_v_rapido = np.abs(fft_v_rapido)[:n_rapido//2]
magnitud_i_rapido = np.abs(fft_i_rapido)[:n_rapido//2]
freq_pos_rapido = frecuencias_rapido[:n_rapido//2]

plt.figure()
plt.plot(freq_pos_rapido, magnitud_i_rapido, color='tab:orange', label='Corriente Lento Real')
plt.title("Visualización espectral del régimen rapido")
plt.xlabel("Tiempo (s)")
plt.ylabel("Corriente (mA)")
plt.show()

#%%
plt.figure()
plt.plot(df_lento.iloc[:, 0], df_lento.iloc[:, 2], color='tab:orange', label='Corriente Lento Real')
plt.title("Visualización directa del régimen Lento")
plt.xlabel("Tiempo (s)")
plt.ylabel("Corriente (mA)")
plt.show()

#%% #%% Observe que los datos de voltaje no son exactamente iguales, ya que el sistema es adaptativo, ajusta el voltaje que inyecta segunpropiedades del sistema que analiza
# Por eso, vamos a analizar si entre las mediciones hay correlacion. 
# Mi señal tiene dos regimenes: uno rapido inivial 0-1.5s y uno lento de 1.5-3s. Separo en dos regimenes.
#Analizo amplitud y desvio estandar

def analizar_segmento(df_segmento):
    amp = (df_segmento.iloc[:, 1].max() - df_segmento.iloc[:, 1].min()) / 2
    std_v = df_segmento.iloc[:, 1].std()
    print(f"Amplitud medida: {amp:.2f} V")
    print(f"Desvío estándar: {std_v:.2f} V")
    return amp, std_v

t_corte= tiempo_corte

df_rapido = df1[df1.iloc[:, 0] < t_corte]
df_lento = df1[df1.iloc[:, 0] >= t_corte]

#%% Ahora puedo analizar mi correinte, ya que mi corriente es COMO reacciona mi señal! Antes de analizarnada vaamos  afiltrarlar, visualmente la señal se ve super limpia y sigue fielmente la naturaleza del voltaje, sin embargo para confirmar feacientemente este hecho, aplicaremos filtros. Intentaremos que esos filtros no modifiquen la señal o la desfacen. 
#%% FILTRADO DIGITAL 
# Hay que filtrar 3 componentes: 
   #- Ruido temrico : ruido de alta frecuencia 
   #- Ruido de línea : 50hz
   #- Ruido electroestatico (deriva de linea) --> En mi espectroy en tiempo la señal esta alineada a 0 no es necesario
# Quiero modificar lo menor posible mi señal por lo que voy a usar un filtro butterworth pasa banda. Vuelvo a hacer el espectrograma para ver si limpio sino aplico notch en la linea 
#%% Diseño filtro
#Al descomponer la señal en ambos segmentos, parece que el pico en 58hrz es un armonico ya que mi señal fundamental de regimen rápido es 19,69
primer_armonico= freq_rapida*3
print(primer_armonico)
#59 APROXIMADAMENTE perfecto es armonico. 
#Quiero un filtro butter para deformar lo menos posible mi señal! La hipotesis es que la señal esta correctamente filtrada por el equipo de medicion 

#%% Diseño de plantilla 
fs = 4000
nyq_frec = fs / 2
ripple = 1          # dB (máxima ondulación en banda de paso)
attenuation = 40    # dB (mínima atenuación en banda de rechazo)
fpass = np.array([1.0, 40.0]) 
fstop = np.array([0.5, 65.0]) 

plt.figure(figsize=(10, 6))
plt.title('Plantilla de Diseño de Filtro Pasabanda')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(True)
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()
#%% Diseño filtro
# Parámetros del filtro
fs = 4000          
orden = 10
lowcut = 1
highcut = 40
sos = signal.butter(orden, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
#%%
w, h = signal.sosfreqz(sos, worN=8000, fs=fs)

# 2. Dibujamos
plt.figure(figsize=(8, 5))
plt.plot(w, 20 * np.log10(np.abs(h) + 1e-15), color='blue', linewidth=2, label='Filtro Butterworth')
# 3. Detalles estéticos y técnicos
plt.title('Respuesta en Frecuencia: Filtro Butterworth (Orden 4)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.xlim(0, 200)       # Ajustá esto según lo que quieras ver
plt.ylim(-60, 5)       # Rango típico para ver bien la atenuación
plt.grid(True, which='both')
plt.legend()
plot_plantilla(filter_type="bandpass", fpass=fpass, ripple=ripple, fstop=fstop, attenuation=attenuation, fs=fs)
plt.show()
#%% Aplico el filtro 

# Aplicar el filtro a la señal idealizada
i_filtrada = signal.sosfiltfilt(sos, df1.iloc[:, 2])
fft_filtrada = np.fft.fft(i_filtrada)
mag_filtrada = np.abs(fft_filtrada)[:n//2]
frecuencias = np.fft.fftfreq(n, d=1/fs)[:n//2]

plt.figure(figsize=(10, 6))
plt.plot(freq_pos , magnitud_i, color='gray', alpha=0.5, label='Original')
plt.plot(frecuencias, mag_filtrada, color='green', label='Filtrada')
plt.title('Comparación: Señal Original vs Filtrada (Régimen Rápido)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.xlim(0, 100) # Zoom en la zona donde está el ruido/armónico
plt.grid(True)
plt.legend()
plt.show()

#%%  Seññal filtrada

plt.figure(figsize=(10, 5))
plt.plot(df1.iloc[:, 0], i_filtrada, color='tab:red', label='Corriente filtrada')
plt.plot(df1.iloc[:, 0], df1.iloc[:, 2], color='tab:blue', label='Corriente sin filtrar')
plt.title("Comparacion antes vs después del filtrado")
plt.xlabel("Tiempo")
plt.ylabel("Corriente (mV)")
plt.legend()
plt.grid(True)
plt.show()

#%% Analisis de corriente 

def analizar_toda_la_señal(df):
    tiempo = df.iloc[:, 0].values
    i = df.iloc[:, 2].values
    v = df.iloc[:, 1].values
    
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
i_peak, i_ss, snr, factor, ruido = analizar_toda_la_señal(df1)
print(f"Pico Promedio: {i_peak:.4f}, SNR: {snr:.2f}, Factor: {factor:.4f}")

#%% Analisis de impedancia 
plt.figure(figsize=(6, 6))
plt.plot(df1.iloc[:, 1], df1.iloc[:, 2], color='purple', alpha=0.5)
plt.title("Corriente vs. Voltaje")
plt.xlabel("Voltaje (V)")
plt.ylabel("Corriente (mA)")
plt.grid(True)
plt.show()