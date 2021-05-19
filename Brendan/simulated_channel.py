import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa
import pyaudio
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import pandas as pd
import bitarray 
import binascii
import math 
from scipy import signal

N = 1024 # DFT size
K = 512 # number of OFDM subcarriers with information
CP = K//8  # length of the cyclic prefix
P = K//8# number of pilot carriers per OFDM block
pilotValue = 1+1j # The known value each pilot transmits

#Define the payload and pilot carriers
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[1::K//P] # Pilots is every (K/P)th carrier.

dataCarriers = np.delete(allCarriers, pilotCarriers)
dataCarriers = np.delete(dataCarriers, 0)

channelResponse = np.array(pd.read_csv("./Brendan/channel.csv", header = None)).ravel() # the impulse response of the wireless channel
H = np.fft.fft(channelResponse, N)

#plt.plot(np.arange(N), abs(H))
#plt.xlabel('FT index'); plt.ylabel('$|H(f)|$'); plt.grid(True); plt.xlim(0, N)
#plt.show()

###IMPORT TEXT FILE FOR TESTING
with open("./Brendan/test_file.txt") as f:
    contents = f.read()

import bitarray
ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))

ba = np.array(ba.tolist())

###IMPORT CHIRP###
def exponential_chirp(T=10.0, f1=60.0, f2=6000.0, window_strength=10.0, fs=44100):
    """Produces chirp and returns impulse characteristics"""

    t_list = np.linspace(0, T, int(round(T * fs)), False)
    profile = []
    r = f2/f1

    # Calculate Sine Sweep time domain values
    for t in t_list:
        value = math.sin(2*math.pi*T*f1*((r**(t/T)-1)/(math.log(r, math.e))))*(1-math.e**(-window_strength*t))*(1-math.e**(window_strength*(t-T)))
        profile.append(value)

    # Format
    profile = np.array(profile)

    return profile
exponentialChirp = exponential_chirp()
#print(exponentialChirp)

#QPSK
mu = 2 # bits per symbol 
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of bits per OFDM symbol

mapping_table = {
    (0,0) : 1+1j,
    (0,1) : -1+1j,
    (1,0) : 1-1j,
    (1,1) : -1-1j
}
demapping_table = {v : k for k, v in mapping_table.items()}

#append required number of zeros to end to send a full final OFDM symbol
ba = np.append(ba, np.zeros(len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)))
bits_SP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))

print(bits_SP.shape[0]) #Number of OFDM symbols

def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

def OFDM_symbol(QPSK_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QPSK_payload  # allocate the data subcarriers
    symbol = np.append(symbol, np.append(0,np.conj(symbol)[:0:-1]))
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def mapToTransmit(bits_SP):
    sound_array = []
    for section in bits_SP:
        QPSK = Mapping(section)
        OFDM_data = OFDM_symbol(QPSK)
        OFDM_time = IDFT(OFDM_data)
        OFDM_withCP = addCP(OFDM_time)
        sound_array.append(OFDM_withCP.real)
    return np.array(sound_array).ravel()

sound = mapToTransmit(bits_SP)
#print(len(sound))

#SIMULATED CHANNEL
def channel(signal, channelResponse, SNRdb=20):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    
    print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

OFDM_TX = sound
OFDM_RX = channel(OFDM_TX, channelResponse)
#plt.figure(figsize=(8,2))
#plt.plot(abs(OFDM_TX), label='TX signal')
#plt.plot(abs(OFDM_RX), label='RX signal')
#plt.legend(fontsize=10)
#plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
#plt.grid(True);
#plt.show()
#print(len(OFDM_RX))

chirp_filtered = channel(exponentialChirp, channelResponse)

#Add some zeros in front to test synchronisation
chirp_filtered = np.append(np.zeros(10000), chirp_filtered)

def chirp_estimator_test(audio, samples=30, f1=60.0, f2=6000.0, T=5, fs=44100, actualH = H):
    x = exponential_chirp(T)
    x_r = x[::-1]

    # Apply exponential envelope and normalise
    for i in range(len(x_r)):
        x_r[i] = x_r[i] * math.e ** (-(i / (T * fs)) * math.log2(f2 / f1))

    x_r = x_r / np.linalg.norm(x_r)

    y = audio / np.linalg.norm(audio)
    # Convolve output with x_r
    h = signal.fftconvolve(x_r, y)

    # Estimate Impulse Response start point
    position = np.where(h == np.amax(h))[0][0]
    
    # Find closest 0 to estimate above and set as Impulse Response start point
    i = 1
    while True:
        if 0 <= h[position - i] <= h[position - i + 1]:
            i += 1
        else:
            break
    position -= i
    if abs(h[position]) > abs(h[position - 1]):
        position -= 1
    h[position] = 0

    # Duplicate and truncate h
    h_0 = h
    h = h[position:int(position + samples)]

    H2 = np.fft.fft(h, N)

    plt.plot(np.linspace(0, len(h) / fs, len(h), False), h)
    plt.title('Measured Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()

    plt.plot(np.linspace(0, len(h_0) / fs, len(h_0), False), h_0)
    plt.title('Full Recording')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()
    
    #plt.semilogy(np.linspace(0, fs, len(H2), False), abs(H2), label = 'measured H')
    plt.plot(np.arange(len(H2)), abs(H2), label = 'measured H')
    plt.plot(np.arange(N), abs(actualH), label = 'actual H')
    plt.xlabel('FT index'); plt.ylabel('$|H(f)|$'); plt.grid(True); plt.xlim(0, N); plt.legend()
    plt.show()
    return h, H2, position

h2, H2, position = chirp_estimator_test(chirp_filtered)

print(position)

def removeCP(signal, a):
    return signal[a:(a+N)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX, N)

def mapToDecode(audio, channelResponse):
    dataArrayEqualized = []
    for i in range(len(audio)//N+CP):
        data = audio[i*(N+CP): (N+CP)*(i+1)]
        data = removeCP(data, CP)
        data = DFT(data)
        Hest = channelResponse
        data_equalized = data/Hest
        dataArrayEqualized.append(data_equalized[1:512][dataCarriers-1])
    return np.array(dataArrayEqualized).ravel()

OFDM_todemap = mapToDecode(OFDM_RX, H)

def Demapping(QPSK):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(QPSK.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

outputdata1, hardDecision = Demapping(OFDM_todemap)
for qpsk, hard in zip(OFDM_todemap[0:500], hardDecision[0:500]):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    plt.plot(hardDecision[0:500].real, hardDecision[0:500].imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Hard Decision demapping');
plt.show()

data1tocsv = outputdata1.ravel()
str1 = ''.join(str(e) for e in data1tocsv)

def text_from_bits(bits, encoding='utf-8', errors='ignore'):
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)

def int2bytes(i):
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))

print(text_from_bits(str1))