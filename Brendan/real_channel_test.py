import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.random import default_rng
from scipy import signal
import binascii
import bitarray
import scipy.interpolate

###IMPORT ORIGINAL TRANSMITED TEXT FILE
with open("./Brendan/test_file.txt") as f:
    contents = f.read()

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))

ba = np.array(ba.tolist())

###IMPORT RECORDED AUDIO###
def audioDataFromFile(filename):
    data, fs = sf.read(filename, dtype='float32')  
    return data

outputData = audioDataFromFile('sound_to_transmit.wav')
receivedData = audioDataFromFile('received_sound.wav')

###OFDM CONFIGS###
fs = 44100
N = 1024 # DFT size
K = 512 # number of OFDM subcarriers with information
CP = K//4  # length of the cyclic prefix
P = K//16# number of pilot carriers per OFDM block
pilotValue = 1+1j # The known value each pilot transmits

#Define the payload and pilot carriers
allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[1::K//P] # Pilots is every (K//P)th carrier.

dataCarriers = np.delete(allCarriers, pilotCarriers)
dataCarriers = np.delete(dataCarriers, 0)

#Define known OFDM symbols configs
seedStart = 2000
numBlocks = 10

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

###SYNCHRONISATION USING CHIRP###
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

    #plt.plot(h)
    #plt.show()
    return position-T*fs

position = chirp_synchroniser(receivedData)
print(position)

#plt.figure(1)
#plt.plot(receivedData)
#plt.vlines(position, color = 'red', ymin = 0, ymax = 1)
#plt.title('Received Data')
#plt.figure(2)
#plt.plot(outputData)
#plt.title('Sent data')
#plt.show()

syncData = receivedData[position + chirpLength*fs:]

###FUNCTIONS REQUIRED FOR DEMODULATION###
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
        dataArrayEqualized.append(data_equalized[1:512][dataCarriers-1])
    return np.array(dataArrayEqualized).ravel()

###CHANNEL ESTIMATION USING KNOWN OFDM SYMBOLS###
def channelEstimateKnownOFDM(knownOFDMBlock, numberOfBlocks = numBlocks, randomSeedStart = seedStart, N = N, CP = CP):
    Hest_at_symbols = np.zeros(N,dtype=complex)

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
    Hest_at_symbols = (Hest_at_symbols * i + receivedSymbols / expectedSymbols) / (i+1) #Averaging over past OFDM blocks

    plt.figure(1)
    plt.plot(np.arange(N), np.log(abs(Hest_at_symbols)), label='Estimated H via cubic interpolation')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.figure(2)
    plt.plot(impulse, label = 'impulse response')
    plt.grid(True); plt.xlabel('samples'); plt.ylabel('h(t)'); plt.legend(fontsize=10)
    plt.show()

    return Hest_at_symbols, impulse
Hest, impulse = channelEstimateKnownOFDM(syncData)

####FINER SYNCHRONISATION USING PILOT TONES####
def findLocationWithPilot(approxLocation, data, rangeOfLocation = 4, pilotValue = pilotValue, pilotCarriers = pilotCarriers):
    """Performs fine synchronization using pilot symbols"""
    minValue = 100000
    for i in range(-rangeOfLocation+1, rangeOfLocation):
        OFDM_symbol = data[approxLocation + i:approxLocation+i+N+CP]
        OFDM_noCP = removeCP(OFDM_symbol, CP)
        OFDM_time = DFT(OFDM_noCP)
        angle_values = np.angle(OFDM_time[pilotCarriers]/pilotValue)
        #plt.plot(pilotCarriers, angle_values, label = i)
        
        absSlope = abs(np.polyfit(pilotCarriers,angle_values,1)[0])

        if absSlope < minValue:
            minValue = absSlope
            bestLocation = i+approxLocation
            print(minValue)
    return bestLocation, minValue

bestLocation, minValue = findLocationWithPilot(numBlocks * (N + CP), syncData)
#plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('Anglular phase'); plt.legend(fontsize=10);
#plt.show()
#print(bestLocation-numBlocks * (N + CP))

#offBySync = bestLocation-numBlocks * (N + CP)
#if offBySync != 0:
#    syncData = receivedData[position+chirpLength*fs+offBySync:]
#    Hest = channelEstimateKnownOFDM(syncData[:numBlocks*(N+CP)])

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

####DEMODULATION####
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

OFDM_todemap = mapToDecode(syncData[numBlocks*(N+CP):], Hest)[:len(ba)//2]

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
for qpsk, hard in zip(OFDM_todemap[0:400], hardDecision[0:400]):
    plt.plot([qpsk.real], [qpsk.imag], 'b-o');
    plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
plt.show()

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

"""
minError = 1
for i in range(-400, 400):
   syncData = receivedData[position + chirpLength*fs+i:]
   #Hest = channelEstimateKnownOFDM(syncData[:numBlocks*(N+CP)])
   OFDM_todemap = mapToDecode(syncData[numBlocks*(N+CP):], Hest)[:len(ba)//2]
   outputdata1, hardDecision = Demapping(OFDM_todemap)
   data1tocsv = outputdata1.ravel()
   demodulatedOutput = ''.join(str(e) for e in data1tocsv)
   error = calculateBER(ba, data1tocsv)
   if error < minError:
       minError = error
       minErrorLocation = i
       print(minError)
   elif i%10 == 0:
       print(i)

print(minError, minErrorLocation)

syncData = receivedData[position + chirpLength*fs+minErrorLocation:]
Hest = channelEstimateKnownOFDM(syncData[:numBlocks*(N+CP)])
OFDM_todemap = mapToDecode(syncData[numBlocks*(N+CP):], Hest)[:len(ba)//2]
outputdata1, hardDecision = Demapping(OFDM_todemap)

for qpsk, hard in zip(OFDM_todemap[0:400], hardDecision[0:400]):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    #plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
plt.show()

data1tocsv = outputdata1.ravel()
demodulatedOutput = ''.join(str(e) for e in data1tocsv)
print(calculateBER(ba, data1tocsv))"""

