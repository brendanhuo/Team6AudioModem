import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa
import pyaudio
import sounddevice as sd
import soundfile as sf
import math, cmath
from scipy.io.wavfile import write

# QPSK Transmitter
class Transmitter:
    def __init__(self, bits_data, N, L, encoding_instructions, max_freq_index=0):
        self.N = N # dft size
        self.L = L # cyclic prefix length
        self.bits_data = bits_data # bits data to be transmitted
        self.mu = 2 # bits per symbol
        self.max_freq_index = max_freq_index
        self.encoding_instructions = encoding_instructions # TODO: may have different encoding of data depending on what it is
        
        self.symbols = []
    
    # convert bits_data to QPSK symbols
    def qpsk(self):
        assert(len(self.bits_data)%self.mu==0)
        self.symbols = []

        for i in range(int(len(self.bits_data) / self.mu)):
            real = 1 / math.sqrt(2)
            imaginary = 1 / math.sqrt(2) * 1j

            index = i*2
            if self.bits_data[index] == '1':
                imaginary *= -1
            
            if self.bits_data[index+1] == '0':
                real *= -1
            
            self.symbols.append(real + imaginary)
        
        self.symbols = np.asarray(self.symbols)
    
    def ofdm(self):
        # Frequency bins that actually contain information, other half is mirrored
        info_block_len = int(self.N/2 - 1)

        if self.max_freq_index != 0:
            max_limit = min(self.max_freq_index, info_block_len)
        else:
            max_limit = info_block_len
        
        ofdm_time = []
        ofdm_freq = []

        index = 0

        while index < len(self.symbols):

            # Symbols in one DFT block of length N - info_block is the DFT block
            info_block = []
            for i in range(max_limit):
                try:
                    info_block.append(self.symbols[index])
                except:
                    break
                index += 1
            
            # Ensure that the information block length is N/2 - 1
            info_padding = info_block_len - len(info_block)
            
            for i in range(info_padding):
                info_block.append(cmath.rect(0.2, 0))
            
            info_block = np.array(info_block)

            # Add the complex conjugates and zero in the beginning and middle to form a OFDM symbol sequence to ensure a real time series
            ofdm_symbol_block = np.concatenate(([0], info_block, [0], info_block[::-1].conjugate()))

            # time domain signal is the iFFT
            time_series_block = np.fft.ifft(ofdm_symbol_block)

            # add cyclic prefix
            if self.L == 0:
                ofdm_time_block = time_series_block
            else:
                ofdm_time_block = np.concatenate((time_series_block[-self.L:], time_series_block))
            
            ofdm_time.append(ofdm_time_block)
            ofdm_freq.append(ofdm_symbol_block)
        
        return np.asarray(ofdm_time), np.asarray(ofdm_freq)


