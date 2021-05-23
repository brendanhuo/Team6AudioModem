from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 

dataCarriers, pilotCarriers = assign_data_pilot(K, P)

### TRANSMITTER ###

# Import text file for testing
with open("./text/lorem.txt") as f:
    contents = f.read()

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())

# Append required number of zeros to end to send a full final OFDM symbol
ba = np.append(ba, np.zeros(len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
print(len(bitsSP))

# Chirp 
exponentialChirp = exponential_chirp(chirp_length)

# OFDM data symbols
sound = map_to_transmit(K, CP, pilotValue, pilotCarriers, dataCarriers, bitsSP)

# Known random OFDM block for channel estimation
knownOFDMBlock = known_ofdm_block(blockNum, seedStart, mu, K, CP, mappingTable)

# Total data sent over channel
dataTotal = np.concatenate((np.zeros(44100), exponentialChirp.ravel(), knownOFDMBlock, sound))

plt.plot(dataTotal)
plt.title("Signal to send")
plt.show()

#write("audio/test_sound_two.wav", fs, dataTotal)

### CHANNEL ###

channelResponse = np.array(pd.read_csv("./channel/channel.csv", header = None)).ravel() 
noiseSNR = 50
ofdmReceived = channel(dataTotal, channelResponse, noiseSNR)
# ofdmReceived = np.append(np.zeros(10000), ofdmReceived)
HChannel = np.fft.fft(channelResponse, N)
# print(len(ofdmReceived))

### RECEIVER ###

# Channel Estimation Test

knownOFDMBlockReceived = channel(knownOFDMBlock, channelResponse, noiseSNR)
hestAtSymbols = channel_estimate_known_ofdm(knownOFDMBlockReceived, seedStart, mappingTable, N, K, CP, mu)

plt.plot(np.arange(N), HChannel, label = 'actual H')
plt.plot(np.arange(N), abs(hestAtSymbols), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

# Symbol Recovery Test
positionChirpStart = chirp_synchroniser(ofdmReceived, chirp_length)

# OFDM block channel estimation
ofdmBlockStart = positionChirpStart + chirp_length * fs
ofdmBlockEnd = positionChirpStart + chirp_length * fs + (N + CP) * blockNum

hest = channel_estimate_known_ofdm(ofdmReceived[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
plt.plot(np.arange(N), HChannel, label = 'actual H')
plt.plot(np.arange(N), abs(hest), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

plt.plot(ofdmReceived[ofdmBlockEnd:])
plt.show()

equalizedSymbols = map_to_decode(ofdmReceived[ofdmBlockEnd:], hest, N, K, CP, dataCarriers)
outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    #plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
plt.show()

dataToCsv = outputData.ravel()
demodulatedOutput = ''.join(str(e) for e in dataToCsv)
print(text_from_bits(demodulatedOutput))

