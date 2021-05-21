import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa
import pyaudio
import sounddevice as sd
import soundfile as sf
import pandas as pd
import bitarray 
import binascii
import math 
import bitarray
from numpy.random import default_rng
from scipy.io.wavfile import write
from scipy import signal


# OFDM
N = 8192 # DFT size
K = N//2 # number of OFDM subcarriers with information
CP = K//4  # length of the cyclic prefix
P = 501 # number of pilot carriers per OFDM block (cannot be a multiple of 2 for some reason)
pilotValue = 1+1j # The known value each pilot transmits

# QPSK
mu = 2 # bits per symbol 
mappingTable = {
    (0,0) : 1+1j,
    (0,1) : -1+1j,
    (1,0) : 1-1j,
    (1,1) : -1-1j
}
demappingTable = {v : k for k, v in mappingTable.items()}

# Chirp
fs = 44100
chirpLength = 1

# OFDM known random symbols seed
seedStart = 2000
blockNum = 10

