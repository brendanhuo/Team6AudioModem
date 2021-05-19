import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa
import pyaudio
import sounddevice as sd
import soundfile as sf
import math, cmath
from scipy.io.wavfile import write

# AWGN channel
class Channel:
    def __init__(self, channel_impulse_response, noise_snr):
        self.channel_impulse_response
        self.noise_snr = noise_snr
    
    def convolve_channel(self, data_transmitted):
        channel_response = []
        for block in data_transmitted:
            channel_response_block = np.convolve(block, self.channel_impulse_response)
            signal_power = np.mean(abs(channel_response_block**2))
            sigma_squared = signal_power * (10 **(-self.noise_snr / 10)) # noise variance
    
            # Generate complex noise with given variance
            noise = np.sqrt(sigma_squared / 2) * (np.random.randn(*channel_response_block.shape)+1j*np.random.randn(*channel_response_block.shape))

            channel_response.append(channel_response_block + noise)
        
        return channel_response


