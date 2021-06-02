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
from binary_utils import *
from ldpc_jossy.py import ldpc

# OFDM
N = 2048 # DFT size
K = N//2 # number of OFDM subcarriers with information
CP = K//4  # length of the cyclic prefix
P = 128 # number of pilot carriers per OFDM block (cannot be a multiple of 2 for some reason)
# pilotValue = (1+1j)/np.sqrt(2) # The known value each pilot transmits
pilotValue = 1+1j # STANDARD

lowerFrequencyBin = 50 # Inclusive
upperFrequencyBin = 700 # Exclusive

QPSK = False
QAM = True

# QPSK
if QPSK:
    mu = 2 # bits per symbol 
    mappingTable = {
        (0,0) : 1+1j,
        (0,1) : -1+1j,
        (1,0) : 1-1j,
        (1,1) : -1-1j
    }
    demappingTable = {v : k for k, v in mappingTable.items()}

# 16 QAM
if QAM:
    mu = 4
    mappingTable = {
        (0,0,0,0) : -3-3j,
        (0,0,0,1) : -1-3j,
        (0,0,1,0) : +3-3j,
        (0,0,1,1) : +1-3j,
        (0,1,0,0) : -3-1j,
        (0,1,0,1) : -1-1j,
        (0,1,1,0) : +3-1j,
        (0,1,1,1) : +1-1j,
        (1,0,0,0) : -3+3j,
        (1,0,0,1) : -1+3j,
        (1,0,1,0) :  3+3j,
        (1,0,1,1) :  1+3j,
        (1,1,0,0) : -3+1j,
        (1,1,0,1) : -1+1j,
        (1,1,1,0) : +3+1j,
        (1,1,1,1) :  1+1j
    }
    demappingTable = {v : k for k, v in mappingTable.items()}

# LDPC
ldpcCoder = ldpc.code()
ldpcBlockLength = ldpcCoder.K

# Chirp Chain
fs = 44100
chirp_length = 1
time_between = 0
number_chirps = 1
window_strength = 50.0 # 10.0 used for a lot of testing
f1 = 100.0 # 60.0 used for a lot of testing
f2 = 10000.0

# Pilot Estimation
pilotImportance = 0.25
pilotValues = True
knownOFDMImportance = 0.25
knownOFDMInData = True

# Chirp Estimation
samples = 10000
smoothing_factor = 1
chirpimportance = 0.5
method_2 = False

# Timings
time_before_data = 0

# OFDM known random symbols seed
# seedStart = 2000 # For a lot of testing files
seedStart = 2021 # STANDARD
blockNum = 10

# Maximum Likelihood Estimation
maximum_likelihood_estimation = False
length_of_MLE = 11

# Bandlimiting
useBandLimit = True

# Blocks pre and post 10 known OFDM
preblocknum = 10
postblocknum = 1

#Known OFDM in data frequency 
knownInDataFreq = 10 #Every 10 data blocks is one known OFDM

# Metadata
metadata = True
multiplier = 1

len_file_len = 32  # bits
num_file_len = 1  # number of repeats

len_file_format = 16  # bits
num_file_format = 1  # number of repeats
file_formats = {"txt" : 1, "tif" : 2, "wav" : 3}

if metadata:
    len_metadata_bits = len_file_len * num_file_len + len_file_format * num_file_format
else:
    len_metadata_bits = 0

# Images
# Assume image shape is just shape of "Autumn"
_, image_shape = image2bits("./image/autumn_small.tif", plot=False)

# Wav
bits_per_sample = 32

# Pilot Values
# allCarriers = np.arange(K)
# pilotCarriers = allCarriers[1::K//P]
# pilotSeed = 2000
# rng = default_rng(pilotSeed)
# bits = rng.binomial(n=1, p=0.5, size=(K*mu))
# bitsSP = bits.reshape(len(bits)//mu, mu)
# symbol = np.array([mappingTable[tuple(b)] for b in bitsSP])
# pilotValue = symbol[pilotCarriers]
