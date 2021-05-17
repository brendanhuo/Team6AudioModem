import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa

N = 1024 # DFT size
K = 512 # number of OFDM subcarriers with information
CP = K//16  # length of the cyclic prefix: 25% of the block
P = K//8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits

allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[1::K//P] # Pilots is every (K/P)th carrier.

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)
dataCarriers = np.delete(dataCarriers, 0)

mu = 2 # bits per symbol 
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of bits per OFDM symbol

mapping_table = {
    (0,0) : 1+1j,
    (0,1) : -1+1j,
    (1,0) : 1-1j,
    (1,1) : -1-1j
}
demapping_table = {v : k for k, v in mapping_table.items()}

#Make random bit stream of length half of possible payload size for future bandpass modulation
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM * 50, ))

bits_SP = bits.reshape((50, len(dataCarriers), mu))

def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])



def OFDM_symbol(QPSK_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QPSK_payload  # allocate the pilot subcarriers
    symbol = np.append(symbol, np.conj(symbol)[::-1])
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

sound_array = []
for section in bits_SP:
    QPSK = Mapping(section)
    OFDM_data = OFDM_symbol(QPSK)
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    sound_array.append(OFDM_withCP.real)

sound_array = np.array(sound_array).ravel()

def play(array, fs=44100):

    audio = array * (2 ** 15 - 1) / np.max(np.abs(array))
    audio = audio.astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

play(sound_array)

