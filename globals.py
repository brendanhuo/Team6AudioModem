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
N = 2048 # DFT size
K = N//2 # number of OFDM subcarriers with information
CP = K//8  # length of the cyclic prefix
P = 129 # number of pilot carriers per OFDM block (cannot be a multiple of 2 for some reason)
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

# Chirp Chain
fs = 44100
chirp_length = 1
time_between = 0
number_chirps = 1
window_strength = 10.0
f1 = 60.0
f2 = 6000.0

# Pilot Estimation
pilotImportance = 0.5
pilotValues = True

# Chirp Estimation
samples = 10000
smoothing_factor = 1
chirpimportance = 0.5
method_2 = False

# Timings
time_before_data = 0

# OFDM known random symbols seed
seedStart = 2000
blockNum = 10

# Maximum Likelihood Estimation
maximum_likelihood_estimation = False
length_of_MLE = 11
