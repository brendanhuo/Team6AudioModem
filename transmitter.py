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
from globals import *

def assign_data_pilot(K, P, bandLimited = True):
    """ Define the data carriers and pilot tone carriers
    # N - the DFT size
    # K - number of OFDM subcarriers with information
    # P - number of pilot tones per OFDM block """

    allCarriers = np.arange(K)
    pilotCarriers = allCarriers[1::K//P]
    dataCarriers = np.delete(allCarriers, pilotCarriers)

    if bandLimited:
        dataCarriersBandlimit = []
        for i in range(len(dataCarriers)):
            if dataCarriers[i] >=lowerFrequencyBin and dataCarriers[i] < upperFrequencyBin:
                dataCarriersBandlimit.append(dataCarriers[i])
        dataCarriers = np.array(dataCarriersBandlimit)
        # dataCarriers = np.delete(dataCarriers, 0)[0:len(dataCarriers)*3//5]
    else:
        dataCarriers = np.delete(dataCarriers, 0)

    return dataCarriers, pilotCarriers


def mapping(bits):
    """ Maps each batch of bits to a constellation symbol"""
    return np.array([mappingTable[tuple(b)] for b in bits])


def ofdm_symbol(K, pilotValue, pilotCarriers, dataCarriers, qpskPayload):
    """Assigns pilot values and payload values to OFDM symbol, take reverse complex conjugate and append to end to make signal passband"""

    # Rather than use zeros, use random bits to fill space that doesn't have pilots or data payloads
    randomBits = np.random.binomial(n=1, p=0.5, size=(mu * K, ))
    randomBits = randomBits.reshape(len(randomBits) // mu, mu)
    symbol = mapping(randomBits)

    # Original version that only used zeros
    #symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    
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
    for section in bitsSP: # For each OFDM block worth of information bits
        qpsk = mapping(section) # Map to constellation values
        ofdmData = ofdm_symbol(K, pilotValue, pilotCarriers, dataCarriers, qpsk) # Create OFDM block with pilot values and data payloads
        ofdmTime = idft(ofdmData) # IFT to time values
        ofdmWithCP = add_cp(CP, ofdmTime) # Add cyclic prefix
        soundArray.append(ofdmWithCP.real)

    return np.array(soundArray).ravel()


def known_ofdm_block(blocknum, randomSeedStart, mu, K, CP, mappingTable):
    """Known OFDM block generation for channel estimation"""
    knownOFDMBlock = []
    rng = default_rng(randomSeedStart) # Standard requires the same OFDM block
    for i in range(blocknum):
        # rng = default_rng(randomSeedStart + i)
        if i == blocknum//2:
            rng = default_rng(randomSeedStart) # Reset random generator so 6-10th blocks are repeat of 1st-5th
        bits = rng.binomial(n=1, p=0.5, size=((K-1)*mu))
        bitsSP = bits.reshape(len(bits)//mu, mu)

        symbol = np.array([mappingTable[tuple(b)] for b in bitsSP])
        ofdmSymbols = np.concatenate(([0], symbol, [0], np.conj(symbol)[::-1]))

        ofdmTime = idft(ofdmSymbols)
        ofdmWithCP = add_cp(CP, ofdmTime)
        knownOFDMBlock.append(ofdmWithCP.real)
        
    return np.array(knownOFDMBlock).ravel()


def append_Metadata(ba, file, lenData):
    """Returns Metadata array to append before file"""

    # Minimum distance encoding

    # File length Data
    step_value = multiplier
    if step_value * lenData > 2**len_file_len:
        raise ValueError("Insufficient memory assinged to Metadata: {} bits required, but only {} allocated".format(step_value * lenData, 2**len_file_len))

    file_len_encoded = bin(step_value * lenData)[2:]
    file_len_encoded = [int(i) for i in str(file_len_encoded)]
    file_len_data = np.concatenate((np.zeros(len_file_len - len(file_len_encoded)).tolist(), file_len_encoded))
    file_len_data = np.tile(file_len_data, num_file_len)
    # print("file length data: ", file_len_data, len(file_len_data))

    # File format Data

    file_format = file_formats[file[-3:]]

    step_value = multiplier
    if step_value * file_format > 2**len_file_format:
        raise ValueError("Insufficient memory assinged to Metadata: {} bits required, but only {} allocated".format(step_value * file_format, 2**len_file_format))

    file_format_encoded = bin(step_value * file_format)[2:]
    file_format_data = np.concatenate((np.zeros(len_file_format - len(file_format_encoded)), np.array([int(i, 2) for i in file_format_encoded])))
    file_format_data = np.tile(file_format_data, num_file_format)
    # print("file format data: ", file_format_data, len(file_format_data))

    metadata = np.concatenate((file_len_data, file_format_data, ba))

    return metadata
