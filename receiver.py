import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa
import pyaudio
import sounddevice as sd
import soundfile as sf
import math, cmath
from scipy.io.wavfile import write

class Receiver:
    def __init__(self, received_symbols, N, L, channel_response, encoding_instructions):
        self.received_symbols = received_symbols
        self.N = N
        self.L = L
        self.encoding_instructions = encoding_instructions 
        self.channel_response = channel_response # TODO: channel response measurement inside this module from received symbols

    def inverse_ofdm(self):
        block_len = self.N + self.L
        received_modulated_symbols = []
        channel_freq_response = np.fft.fft(self.channel_response, n=self.N)

        # iterate through ofdm time blocks
        for block in self.received_symbols:
            # Remove cyclic prefix
            block_data = block[self.L:]
            
            freq_content = np.fft.fft(block_data)

            # Equalization
            equalized_symbols = np.divide(np.asarray(block_data), freq_content)[1:self.N//2]

            received_modulated_symbols.append(received_modulated_symbols)
        
        return received_modulated_symbols
    
    def decode_symbols(self, received_symbols):
        # QPSK
        decoded_bits = []
        for block in received_symbols:
            block_bits = []
            for symbol in block:
                real_part = symbol.real
                imag_part = symbol.imag

                if real_part > 0 and imag_part > 0:
                    block_bits.append(0)
                    block_bits.append(0)
                elif symbol.real < 0 and symbol.imag > 0:
                    block_bits.append(0)
                    block_bits.append(1)
                elif symbol.real < 0 and symbol.imag < 0:
                    block_bits.append(1)
                    block_bits.append(1)
                elif symbol.real > 0 and symbol.imag < 0:
                    block_bits.append(1)
                    block_bits.append(0)
            decoded_bits.append(block_bits)
        
        return np.asarray(decoded_bits).flatten()














