import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa

N = 1024 # DFT size
K = 512 # number of OFDM subcarriers with information
CP = K//8  # length of the cyclic prefix: 25% of the block
P = K//8# number of pilot carriers per OFDM block
pilotValue = 1+1j # The known value each pilot transmits

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
    symbol[dataCarriers] = QPSK_payload  # allocate the data subcarriers
    symbol = np.append(symbol, np.append(0,np.conj(symbol)[:0:-1]))
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
import pyaudio
import sounddevice as sd
from scipy.io.wavfile import write

def play(array, fs=44100):

    audio = array * (2 ** 15 - 1) / np.max(np.abs(array))
    audio = audio.astype(np.float32)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

def playrec(array, fs = 44100):
    audio = array * (2 ** 15 - 1) / np.max(np.abs(array))
    audio = audio.astype(np.float32)
    myrecording = sd.playrec(audio, fs, channels=1)
    print("recording")
    print(sd.default.device)
    sd.wait()  # Wait until recording is finished
    print("writing")
    write('test_recording.wav', fs, myrecording)  # Save as WAV file 
    print("done")
    return myrecording

def sound(array, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=len(array.shape), rate=fs, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

#playrec(sound_array)

audio = sound_array * (2 ** 15 - 1) / np.max(np.abs(sound_array))
audio = sound_array.astype(np.float32)
print("writing")
write('test_speaker_info.wav', 44100, audio)  # Save as WAV file 
print("done")

#plt.plot(np.arange(len(audio)), audio)
#plt.show()

audio = np.append(np.zeros(6790), audio)
corrArray = []
maximum = 0
count = 0
corrlast = 0
countLimit = CP//2
for i in range(20000):
    corr = np.correlate(audio[i:i+CP], audio[i+N:i+N+CP]/CP)
    corrArray.append(corr)
    if corr>corrlast:
        count += 1
        corrlast = corr
    else:
        if count > countLimit:
            location = i-1
            break
        else:
            corrlast = corr
            count = 0
     
#plt.plot(np.arange(len(corrArray[6700:6900])), corrArray[6700:6900])
#plt.show()
print(location)

def removeCP(signal, a):
    return signal[a:(a+N)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX, N)

def findLocation(location, data, rangeOfLocation = 3, pilotValue = 1+1j, pilotCarriers = pilotCarriers):
    minValue = 100000
    for i in range(-rangeOfLocation, rangeOfLocation):
        OFDM_symbol = data[location + i:location+i+N+CP]
        OFDM_noCP = removeCP(OFDM_symbol, CP)
        OFDM_time = DFT(OFDM_noCP)
        totalValue = np.sum(abs(OFDM_time[np.append(pilotCarriers, -pilotCarriers)]*np.conj(pilotValue).imag))
        print(totalValue)
        if totalValue < minValue:
            minValue = totalValue
            bestLocation = i+location
            print(minValue)
    return bestLocation, minValue

bestLocation, minValue = findLocation(location, audio)
print(bestLocation)
array2 = []
def removeZeros(position,data):
    return data[position:]

data = removeZeros(bestLocation, audio)

for i in range(len(audio)//1056):
    data = audio[i*1056: 1056*(i+1)]
    data = removeCP(data, CP)
    data = DFT(data)
    array2.append(data[1:512][dataCarriers-1]) #first value and 512th value are 0, 513-1023 are conjugate of 1-511 so do not hold any information
    
array2 = np.array(array2).ravel()

todisplay = array2[:511-P]

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

outputdata1, hardDecision = Demapping(todisplay)
for qpsk, hard in zip(todisplay, hardDecision):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    plt.plot(hardDecision.real, hardDecision.imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Hard Decision demapping');
#plt.show()

#print(todisplay)