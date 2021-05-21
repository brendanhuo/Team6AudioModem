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
import bitarray
from numpy.random import default_rng


N = 4096# DFT size
K = N//2 # number of OFDM subcarriers with information
CP = K//4  # length of the cyclic prefix
P = K//12# number of pilot carriers per OFDM block
pilotValue = 1+1j # The known value each pilot transmits

#Define the payload and pilot carriers
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[1::K//P] # Pilots is every (K//P)th carrier.

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

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))

ba = np.array(ba.tolist())

###IMPORT CHIRP###
chirpLength = 1
def exponential_chirp(T=chirpLength, f1=60.0, f2=6000.0, window_strength=10.0, fs=44100):
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

#print(bits_SP.shape[0]) #Number of OFDM symbols

def Mapping(bits):
    """Maps each batch of bits to a constellation symbol"""
    return np.array([mapping_table[tuple(b)] for b in bits])

def OFDM_symbol(QPSK_payload):
    """Assigns pilot values and payload values to OFDM symbol, take reverse complex conjugate and append to end to make signal passband"""
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QPSK_payload  # allocate the data subcarriers
    symbol = np.append(symbol, np.append(0,np.conj(symbol)[:0:-1]))
    return symbol


def IDFT(OFDM_data):
    """Take IDFT"""
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    """Adds cyclic prefix"""
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning

def mapToTransmit(bits_SP):
    """Create full sequence to transmit through speakers"""
    sound_array = []
    for section in bits_SP:
        QPSK = Mapping(section)
        OFDM_data = OFDM_symbol(QPSK)
        OFDM_time = IDFT(OFDM_data)
        OFDM_withCP = addCP(OFDM_time)
        sound_array.append(OFDM_withCP.real)
    return np.array(sound_array).ravel()

#####KNOWN OFDM SYMBOLS GENERATION####    
seedStart = 2000
def knownOFDMBlock(blocknum = 30, randomSeedStart = seedStart, K = K):
    knownOFDMBlock = []
    for i in range(blocknum):
        rng = default_rng(randomSeedStart + i)
        bits = rng.binomial(n=1, p=0.5, size=((K-1)*2))
        bits_SP = bits.reshape(len(bits)//mu, mu)
        symbol = np.array([mapping_table[tuple(b)] for b in bits_SP])
        symbol = np.append(np.append(0, symbol), np.append(0,np.conj(symbol)[::-1]))
        OFDM_time = IDFT(symbol)
        OFDM_withCP = addCP(OFDM_time)
        knownOFDMBlock.append(OFDM_withCP.real)
        
    return np.array(knownOFDMBlock).ravel()

knownOFDMBlock = knownOFDMBlock()

    
sound = mapToTransmit(bits_SP)
#print(len(sound))

####Combine chirp, known OFDM symbols, and OFDM data blocks to get total sound to transmit
toSendOverActualChannel = np.concatenate((np.zeros(44100),exponentialChirp.ravel(), knownOFDMBlock, sound, np.zeros(44100)))

plt.plot(toSendOverActualChannel)
plt.show()
write('sound_to_transmit.wav', 44100, toSendOverActualChannel)

#SIMULATED CHANNEL
def channel(signal, channelResponse, SNRdb=25):
    """Creates a simulated channel with given channel impulse response and white Gaussian noise"""
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    
    #print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise

OFDM_TX = sound
OFDM_RX = channel(OFDM_TX, channelResponse)
OFDM_RX = np.append(np.zeros(10000), OFDM_RX)


knownOFDMBlock_RX = channel(knownOFDMBlock, channelResponse)

#plt.plot(np.arange(len(OFDM_RX)), OFDM_RX)
#plt.show()


####CHANNEL ESTIMATION USING KNOWN OFDM SYMBOLS#####
def channelEstimateKnownOFDM(knownOFDMBlock, randomSeedStart = seedStart, N = N, CP = CP):
    numberOfBlocks = len(knownOFDMBlock) // (N+CP)

    Hest_at_symbols = np.zeros(N, dtype = complex)

    for i in range(numberOfBlocks):
        rng = default_rng(randomSeedStart + i)
        bits = rng.binomial(n=1, p=0.5, size=((K-1)*2))
        bits_SP = bits.reshape(len(bits)//mu, mu)
        symbol = np.array([mapping_table[tuple(b)] for b in bits_SP])
        expectedSymbols = np.append(np.append(0, symbol), np.append(0,np.conj(symbol)[::-1]))

        receivedSymbols = knownOFDMBlock[i*(N+CP):(i+1)*(N+CP)]
        receivedSymbols = removeCP(receivedSymbols, CP)
        receivedSymbols = DFT(receivedSymbols)
            
        for j in range(N):
            if j == N//2 or j == 0:
                Hest_at_symbols[j] == 0
            else:
                div = (receivedSymbols[j]/expectedSymbols[j] )
                Hest_at_symbols[j] = (Hest_at_symbols[j] * i + div) / (i + 1) #Average over past OFDM blocks
                    
    impulse = np.fft.ifft(Hest_at_symbols)
    
    plt.figure(1)
    plt.plot(np.arange(N), H, label = 'actual H')
    plt.plot(np.arange(N), abs(Hest_at_symbols), label='Estimated H via known OFDM')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.figure(2)
    plt.plot(impulse[0:30], label = 'measured impulse')
    #plt.plot(channelResponse, label = 'actual impulse')
    plt.grid(True); plt.xlabel('samples'); plt.ylabel('h(t)'); plt.legend(fontsize=10)
    plt.show()

    return Hest_at_symbols, impulse

####CHANNEL ESTIMATION USING PILOT SYMBOLS####
def channelEstimate(OFDM_demod):
    """Performs channel estimation using pilot values"""
    pilots_pos = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    pilots_neg = OFDM_demod[-pilotCarriers]
    Hest_at_pilots1 = pilots_pos / pilotValue # divide by the transmitted pilot values
    Hest_at_pilots2 = pilots_neg / np.conj(pilotValue)
    
    Hest_at_pilots = np.append(Hest_at_pilots1, Hest_at_pilots2).ravel()
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(np.append(pilotCarriers, N-pilotCarriers).ravel(), abs(Hest_at_pilots), kind='cubic', fill_value="extrapolate")(np.arange(N))
    Hest_phase = scipy.interpolate.interp1d(np.append(pilotCarriers, N-pilotCarriers).ravel(), np.angle(Hest_at_pilots), kind='cubic', fill_value="extrapolate")(np.arange(N))
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    
    #lt.plot(np.arange(N), H, label = 'actual H')
    #plt.stem(np.append(pilotCarriers, N-pilotCarriers), abs(Hest_at_pilots), label='Pilot estimates')
    #plt.plot(np.arange(N), abs(Hest), label='Estimated H via cubic interpolation')
    #plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    #plt.show()
    return Hest

def removeCP(signal, a):
    """Removes cyclic prefix"""
    return signal[a:(a+N)]

def DFT(OFDM_RX):
    """Takes DFT"""
    return np.fft.fft(OFDM_RX, N)

def mapToDecode(audio, channelH):
    """Builds demodulated constellation symbol from OFDM symbols"""
    dataArrayEqualized = []
    for i in range(len(audio)//N+CP):
        data = audio[i*(N+CP): (N+CP)*(i+1)]
        data = removeCP(data, CP)
        data = DFT(data)
        #Hest = channelEstimate(data)
        Hest = channelH
        data_equalized = data/Hest
        dataArrayEqualized.append(data_equalized[1:K][dataCarriers-1])
    return np.array(dataArrayEqualized).ravel()

####Channel Estimation using known OFDM symbols####
Hest_known_symbols, impulse = channelEstimateKnownOFDM(knownOFDMBlock_RX)

#####THIS DOES NOT WORK WELL, USE CHIRP TO SYNCHRONISE INSTEAD####
def CPsync(audio, limit = 15000, CP = CP):
    """Synchronization using cyclic prefix"""
    corrArray = []
    #count = 0
    #corrlast = 0
    #countLimit = CP//4
    for i in range(limit):
        corr = np.correlate(audio[i:i+CP], np.conj(audio[i+N:i+N+CP]))/CP
        corrArray.append(corr)
        '''if corr>corrlast:
            count += 1
            corrlast = corr
        else:
            if count > countLimit:
                location = i-1
                break
            else:
                corrlast = corr
                count = 0'''
    corrArray = np.array(corrArray).ravel()
    indexes = scipy.signal.argrelextrema(
    corrArray,
    comparator=np.greater,order=2
)
    return indexes, corrArray

peaks, corrArray = CPsync(OFDM_RX) 
#plt.plot(corrArray)
#plt.show()
#print(peaks)

####SYNCHRONISATION USING PILOT SYMBOls####
def findLocationWithPilot(approxLocation, data, rangeOfLocation = 4, pilotValue = pilotValue, pilotCarriers = pilotCarriers):
    """Performs fine synchronization using pilot symbols"""
    minValue = 100000
    for i in range(-rangeOfLocation+1, rangeOfLocation):
        OFDM_symbol = data[approxLocation + i:approxLocation+i+N+CP]
        OFDM_noCP = removeCP(OFDM_symbol, CP)
        OFDM_time = DFT(OFDM_noCP)
        angle_values = np.angle(OFDM_time[pilotCarriers]/pilotValue)
        plt.plot(pilotCarriers, angle_values, label = i)
        
        absSlope = abs(np.polyfit(pilotCarriers,angle_values,1)[0])

        if absSlope < minValue:
            minValue = absSlope
            bestLocation = i+approxLocation
            #print(minValue)
    return bestLocation, minValue

bestLocation, minValue = findLocationWithPilot(10000, OFDM_RX)
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('Anglular phase'); plt.legend(fontsize=10);
plt.show()
print(bestLocation)

def removeZeros(position,data):
    """Removes preappended zeros from data"""
    return data[position:]

OFDM_RX = removeZeros(bestLocation, OFDM_RX)

#plt.figure(figsize=(8,2))
#plt.plot(abs(OFDM_TX), label='TX signal')
#plt.plot(abs(OFDM_RX), label='RX signal')
#plt.legend(fontsize=10)
#plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
#plt.grid(True);
#plt.show()
#print(len(OFDM_RX))


####CHIRP STUFF####
chirp_filtered = channel(exponentialChirp, channelResponse)

#Add some zeros in front to test synchronisation
chirp_filtered = np.append(np.zeros(10000), chirp_filtered)

#Synchronisation using chirp
def chirp_synchroniser(audio, f1=60.0, f2=6000.0, T=chirpLength, fs=44100):
    """Tries to estimate impulse response and perform gross synchronization using a chirp"""
    x = exponential_chirp(T)
    x_r = x[::-1]

    # Apply exponential envelope and normalise
    for i in range(len(x_r)):
        x_r[i] = x_r[i] * math.e ** (-(i / (T * fs)) * math.log2(f2 / f1))

    x_r = x_r / np.linalg.norm(x_r)

    # Format and normalise
    y = audio
    y = y / np.linalg.norm(y)

    # Convolve output with x_r
    h = signal.fftconvolve(x_r, y)

    # Estimate Impulse Response start point
    position = np.where(h == np.amax(h))[0][0]

    return position-T*fs

def chirp_test(audio, samples=2000, f1=60.0, f2=6000.0, T=chirpLength, fs=44100, channelH = H, N = N):
    # Create chirp, and time reversed signal x_r
    x = exponential_chirp(T)
    x_r = x[::-1]

    # Apply exponential envelope and normalise
    for i in range(len(x_r)):
        x_r[i] = x_r[i] * math.e ** (-(i / (T * fs)) * math.log2(f2 / f1))

    x_r = x_r / np.linalg.norm(x_r)

    # Format and normalise
    y = audio
    y = y / np.linalg.norm(y)

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
    #position -= i
    if abs(h[position]) > abs(h[position - 1]):
        position -= 1
    h[position] = 0

    # Duplicate and truncate h
    h_0 = h
    h = h[position:int(position + samples)]

    # View results

    plt.plot(np.linspace(0, len(h) / fs, len(h), False), h)
    plt.title('Measured Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()

    # Find Frequency Response
    H = np.fft.fft(h, N)
    plt.plot(abs(H), label = 'chirp estimated H')
    plt.plot(abs(channelH), label = 'actual H')
    plt.grid(True)
    plt.legend()
    plt.show()

    return h, H, position-fs*T
position = chirp_synchroniser(chirp_filtered)
print(position)
####END OF CHIRP STUFF####

OFDM_todemap = mapToDecode(OFDM_RX, H)

def Demapping(QPSK):
    """Demaps from demodulated constellation to original bit sequence"""
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
#for qpsk, hard in zip(OFDM_todemap[0:400], hardDecision[0:400]):
#    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
#    #plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
#plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
#plt.show()

data1tocsv = outputdata1.ravel()
demodulatedOutput = ''.join(str(e) for e in data1tocsv)

def text_from_bits(bits, encoding='utf-8', errors='ignore'):
    """Convert byte sequence to text"""
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)

def int2bytes(i):
    """Converts bit stream to bytes"""
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))

def calculateBER(ba, audioOutput):
    """Calculates bit error rate"""
    errorCount = 0
    for i in range(len(ba)):
        if ba[i] != audioOutput[i]:
            errorCount += 1
    return errorCount / len(ba)

print(calculateBER(ba, data1tocsv))

print(text_from_bits(demodulatedOutput))

