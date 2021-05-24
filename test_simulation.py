from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from audio_utils import *
from ldpc_jossy.py import ldpc

dataCarriers, pilotCarriers = assign_data_pilot(K, P)

### TRANSMITTER ###

# Import text file for testing
with open("./text/lorem.txt") as f:
    contents = f.read()

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())

# LDPC coding
ldpcCoder = ldpc.code(standard='802.16', rate='1/2', z=27)
ldpcBlockLength = ldpcCoder.K # determined by LDPC code

# u = np.random.randint(0,2,ldpcCoder.K)
# print(ldpcCoder.encode(u))

# Pad ba for ldpc
ba = np.append(ba, np.zeros(((len(ba)) // ldpcBlockLength + 1) * ldpcBlockLength - len(ba)))
ba = np.reshape(ba, (-1, ldpcBlockLength))
ldpcConvert = []
for i in range(len(ba)):
    encoded = ldpcCoder.encode(ba[i])
    ldpcConvert.append(encoded)

ba = np.array(ldpcConvert).ravel()

# Append required number of zeros to end to send a full final OFDM symbol
numZerosAppended = len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)
ba = np.append(ba, np.zeros(numZerosAppended))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
# print(bitsSP.shape)

# Chirp 
# exponentialChirp = exponential_chirp(chirpLength)
exponentialChirp = exponential_chirp_chain()
# OFDM data symbols
sound = map_to_transmit(K, CP, pilotValue, pilotCarriers, dataCarriers, bitsSP)
# sound = sound / np.max(sound)

# Known random OFDM block for channel estimation
knownOFDMBlock = known_ofdm_block(blockNum, seedStart, mu, K, CP, mappingTable)
# knownOFDMBlock = knownOFDMBlock / np.max(knownOFDMBlock)

# Total data sent over channel
# dataTotal = np.concatenate((np.zeros(fs), exponentialChirp.ravel(), (np.zeros(fs * time_before_data)), knownOFDMBlock, sound))
dataTotal = np.concatenate((np.zeros(fs), exponentialChirp.ravel(), knownOFDMBlock, sound))
print(len(sound) / (CP + N))

save(dataTotal, "audio/chirp_chain.wav")

plt.plot(dataTotal)
plt.title("Signal to send")
plt.show()

# write("audio/chirp_chain.wav", fs, dataTotal)

### CHANNEL ###

channelResponse = np.array(pd.read_csv("./channel/channel.csv", header = None)).ravel() 
noiseSNR = 100
ofdmReceived = channel(dataTotal, channelResponse, noiseSNR)
# ofdmReceived = np.append(np.zeros(10000), ofdmReceived)
HChannel = np.fft.fft(channelResponse, N)
# print(len(ofdmReceived))

### RECEIVER ###

# Channel Estimation Test

knownOFDMBlockReceived = channel(knownOFDMBlock, channelResponse, noiseSNR)
hestAtSymbols, noiseEstimates = channel_estimate_known_ofdm(knownOFDMBlockReceived, seedStart, mappingTable, N, K, CP, mu)

plt.plot(np.arange(N), HChannel, label = 'actual H')
plt.plot(np.arange(N), abs(hestAtSymbols), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

noiseEstimateReal = np.mean(noiseEstimates.real ** 2)
noiseEstimateImag = np.mean(noiseEstimates.imag ** 2)

print("Estimated real part of noise: " + str(noiseEstimateReal))
print("Estimated imaginary part of noise: " + str(noiseEstimateImag))

# Symbol Recovery Test
positionChirpEnd = chirp_synchroniser(ofdmReceived)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd
ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum

hest, noiseEstimates = channel_estimate_known_ofdm(ofdmReceived[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
plt.plot(np.arange(N), HChannel, label = 'actual H')
plt.plot(np.arange(N), abs(hest), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

plt.plot(ofdmReceived[ofdmBlockEnd:])
plt.show()

equalizedSymbols, hestAll = map_to_decode(ofdmReceived[ofdmBlockEnd:], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue)
print(equalizedSymbols[:100])
llrsReceived = return_llrs(equalizedSymbols, hestAll, (noiseEstimateReal, noiseEstimateImag))[:-numZerosAppended]

llrsReceived = np.reshape(llrsReceived, (-1, 2 * ldpcCoder.K))
outputData = []
for block in llrsReceived:
    ldpcDecode, _ = ldpcCoder.decode(block)
    outputData.append(ldpcDecode)
outputData = np.array(outputData).ravel()
np.place(outputData, outputData>0, 0)
np.place(outputData, outputData<0, 1)

# outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

# for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
#     plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
#     plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
# plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
# plt.show()

dataToCsv = np.array(outputData, dtype=int)
demodulatedOutput = ''.join(str(e) for e in dataToCsv)
print(text_from_bits(demodulatedOutput))

