import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa
import pyaudio
import sounddevice as sd
import soundfile as sf
import math, cmath
from scipy.io.wavfile import write
from numpy.random import default_rng


def assign_data_pilot(K, P, bandLimited = False):
    """ Define the data carriers and pilot tone carriers
    # N - the DFT size
    # K - number of OFDM subcarriers with information
    # P - number of pilot tones per OFDM block """

    allCarriers = np.arange(K)
    pilotCarriers = allCarriers[1::K//P]
    dataCarriers = np.delete(allCarriers, pilotCarriers)

    if bandLimited:
        dataCarriers = np.delete(dataCarriers, 0)[:len(dataCarriers)*3//4]
    else:
        dataCarriers = np.delete(dataCarriers, 0)

    return dataCarriers, pilotCarriers


def mapping(bits):
    """ Maps each batch of bits to a constellation symbol"""

    mapping_table = {
        (0,0) : (1+1j) / np.sqrt(2),
        (0,1) : (-1+1j) / np.sqrt(2),
        (1,0) : (1-1j) / np.sqrt(2),
        (1,1) : (-1-1j) / np.sqrt(2)
    }
    demapping_table = {v : k for k, v in mapping_table.items()}

    return np.array([mapping_table[tuple(b)] for b in bits])


def ofdm_symbol(K, pilotValue, pilotCarriers, dataCarriers, qpskPayload):
    """Assigns pilot values and payload values to OFDM symbol, take reverse complex conjugate and append to end to make signal passband"""

    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = qpskPayload  # allocate the data subcarriers
    symbol = np.append(symbol, np.append(0, np.conj(symbol)[:0:-1]))

    return symbol


def idft(ofdmData):
    """Take IDFT"""

    return np.fft.ifft(ofdmData)


def add_cp(CP, ofdmTime):
    """Adds cyclic prefix"""

    cp = ofdmTime[-CP:]               # take the last CP samples ...

    return np.hstack([cp, ofdmTime])  # ... and add them to the beginning


def map_to_transmit(K, CP, pilotValue, pilotCarriers, dataCarriers, bitsSP):
    """Create full sequence to transmit through speakers"""

    soundArray = []
    for section in bitsSP:
        qpsk = mapping(section)
        ofdmData = ofdm_symbol(K, pilotValue, pilotCarriers, dataCarriers, qpsk)
        ofdmTime = idft(ofdmData)
        ofdmWithCP = add_cp(CP, ofdmTime)
        soundArray.append(ofdmWithCP.real)

    return np.array(soundArray).ravel()


def known_ofdm_block(blocknum, randomSeedStart, mu, K, CP, mappingTable):
    """Known OFDM block generation for channel estimation"""

    knownOFDMBlock = []
    for i in range(blocknum):
        rng = default_rng(randomSeedStart + i)
        bits = rng.binomial(n=1, p=0.5, size=((K-1)*2))
        bitsSP = bits.reshape(len(bits)//mu, mu)

        symbol = np.array([mappingTable[tuple(b)] for b in bitsSP])
        ofdmSymbols = np.append(np.append(0, symbol), np.append(0,np.conj(symbol)[::-1]))

        ofdmTime = idft(ofdmSymbols)
        ofdmWithCP = add_cp(CP, ofdmTime)
        knownOFDMBlock.append(ofdmWithCP.real)
        
    return np.array(knownOFDMBlock).ravel()
