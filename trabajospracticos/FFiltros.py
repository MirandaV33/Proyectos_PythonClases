# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:12:10 2025

@author: l
"""

import numpy as np
import scipy.signal as sig
import scipy.io as sio


def crear_filtro_pasabajos(fs, fpass, fstop, metodo, numtaps):
    nyq = fs / 2
    if metodo == 'ventana':
        return sig.firwin(numtaps, fpass, window='hamming', fs=fs, pass_zero=True)
    elif metodo == 'cuadrados minimos':
        freq= [0, fpass, fstop, nyq]
        gain= [1, 1, 0, 0]
        return sig.firls(numtaps, freq, gain,  fs=fs)

def crear_filtro_pasaltos(fs, fpass, fstop, metodo, numtaps):
    nyq = fs / 2
    if metodo == 'ventana':
        return sig.firwin(numtaps, fpass, window='hamming', fs=fs, pass_zero=False)
    elif metodo == 'cuadrados minimos':
        freq= [0, fstop, fpass, nyq]
        gain= [0, 0, 1, 1]
        return sig.firls(numtaps, freq, gain,  fs=fs)

def concatenar_filtros(signal, filtro1, filtro2):
    salida1 = np.convolve(signal, filtro1, mode='same')
    salida2 = np.convolve(salida1, filtro2, mode='same')
    return salida2



