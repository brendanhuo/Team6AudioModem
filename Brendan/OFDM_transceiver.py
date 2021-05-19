import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa
import pyaudio
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf

####Configs######
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

def mapToTransmit(bits_SP):
    sound_array = []
    for section in bits_SP:
        QPSK = Mapping(section)
        OFDM_data = OFDM_symbol(QPSK)
        OFDM_time = IDFT(OFDM_data)
        OFDM_withCP = addCP(OFDM_time)
        sound_array.append(OFDM_withCP.real)
    return np.array(sound_array).ravel()

sound_array = mapToTransmit(bits_SP)

###Some Audio Functions###

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

def record(seconds):
    myrecording = sd.rec(int(seconds * sd.default.samplerate))
    print("recording")
    print(sd.default.device)
    sd.wait()  # Wait until recording is finished
    print("done")
    return myrecording

def audioDataFromFile(filename):
    data, fs = sf.read(filename, dtype='float32')  
    return data

#playrec(sound_array)

#Save random sequence to wav
audio = sound_array.astype(np.float32)
print("writing")
write('test_speaker_info.wav', 44100, audio)  # Save as WAV file 
print("done")
print(np.max(audio))
#plt.plot(np.arange(len(audio)), audio)
#plt.show()

#Import back data that you just made wav file from
audio = audioDataFromFile('test_speaker_info.wav')

#append some zeros infront to simulate some simplistic synchronisation issue
audio = np.append(np.zeros(6790), audio)

def CPsync(audio, limit = 10000, countLimit = CP//2, CP = CP):
    corrArray = []
    count = 0
    corrlast = 0
    countLimit = CP//2
    for i in range(limit):
        corr = np.correlate(audio[i:i+CP], np.conj(audio[i+N:i+N+CP]))/CP
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
    return location, corrArray
location, corrarray = CPsync(audio) 
#plt.plot(np.arange(len(corrArray[6700:6900])), corrArray[6700:6900])
#plt.show()
print(location)

def removeCP(signal, a):
    return signal[a:(a+N)]

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX, N)

def findLocationWithPilot(location, data, rangeOfLocation = 5, pilotValue = pilotValue, pilotCarriers = pilotCarriers):
    minValue = 100000
    for i in range(-rangeOfLocation+1, rangeOfLocation):
        OFDM_symbol = data[location + i:location+i+N+CP]
        OFDM_noCP = removeCP(OFDM_symbol, CP)
        OFDM_time = DFT(OFDM_noCP)
        angle_values = np.angle(OFDM_time[pilotCarriers]/pilotValue)
        plt.plot(pilotCarriers, angle_values, label = i)
        
        absSlope = abs(np.polyfit(pilotCarriers,angle_values,1)[0])

        if absSlope < minValue:
            minValue = absSlope
            bestLocation = i+location
            print(minValue)
    return bestLocation, minValue

bestLocation, minValue = findLocationWithPilot(location, audio)
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('Anglular phase'); plt.legend(fontsize=10)
plt.show()
print(bestLocation)

def removeZeros(position,data):
    return data[position:]

audio = removeZeros(bestLocation, audio)


def channelEstimate(OFDM_demod):
    pilots_pos = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    pilots_neg = OFDM_demod[-pilotCarriers]
    Hest_at_pilots1 = pilots_pos / pilotValue # divide by the transmitted pilot values
    Hest_at_pilots2 = pilots_neg / np.conj(pilotValue)
    
    Hest_at_pilots = np.append(Hest_at_pilots1, Hest_at_pilots2)
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    Hest_abs = scipy.interpolate.interp1d(np.append(pilotCarriers, N-pilotCarriers), abs(Hest_at_pilots), kind='linear', fill_value="extrapolate")(np.arange(N))
    Hest_phase = scipy.interpolate.interp1d(np.append(pilotCarriers, N-pilotCarriers), np.angle(Hest_at_pilots), kind='linear', fill_value="extrapolate")(np.arange(N))
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    
    #plt.stem(np.append(pilotCarriers, N-pilotCarriers), abs(Hest_at_pilots), label='Pilot estimates')
    #plt.plot(np.arange(N), abs(Hest), label='Estimated channel via interpolation')
    #plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    #plt.ylim(0,2)
    #plt.show()
    return Hest


dataArray = []
dataArrayEqualized = []
for i in range(len(audio)//1056):
    data = audio[i*1056: 1056*(i+1)]
    data = removeCP(data, CP)
    data = DFT(data)
    Hest = channelEstimate(data)
    data_equalized = data/Hest
    dataArray.append(data[1:512][dataCarriers-1]) #first value and 512th value are 0, 513-1023 are conjugate of 1-511 so do not hold any information
    dataArrayEqualized.append(data_equalized[1:512][dataCarriers-1])

dataArray = np.array(dataArray).ravel()
dataArrayEqualized = np.array(dataArrayEqualized).ravel()

todisplay = dataArray[:511-P]
todisplayEqualized = dataArrayEqualized[:511-P]
print(len(todisplay))

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
plt.show()

outputdata2, hardDecision2 = Demapping(todisplayEqualized)
for qpsk, hard in zip(todisplayEqualized, hardDecision2):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    plt.plot(hardDecision2.real, hardDecision2.imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Hard Decision demapping');
#plt.show()

