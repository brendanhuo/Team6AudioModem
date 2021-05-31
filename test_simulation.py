from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from audio_utils import *
from ldpc_jossy.py import ldpc

dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = True)

### TRANSMITTER ###

# Import text file for testing
with open("./text/asyoulik.txt") as f:
    contents = f.read()
    contents = contents[:len(contents)//8]

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())
actualData = ba

lenData = len(ba)

# LDPC encoding

ldpcCoder = ldpc.code()
ldpcBlockLength = ldpcCoder.K

# Pad ba for ldpc
ba = np.append(ba, np.zeros(((len(ba)) // ldpcBlockLength + 1) * ldpcBlockLength - len(ba)))
ba = np.reshape(ba, (-1, ldpcBlockLength))
ldpcConvert = []
for i in range(len(ba)):
    encoded = ldpcCoder.encode(ba[i])
    ldpcConvert.append(encoded)

ba = np.array(ldpcConvert).ravel()

# Append required number of zeros to end to send a full final OFDM symbol
numZerosAppend = len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)
ba = np.append(ba, np.random.binomial(n=1, p=0.5, size=(numZerosAppend, )))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))

numOFDMblocks = len(bitsSP)
print(numOFDMblocks)

# Chirp 
exponentialChirp = exponential_chirp()
# exponentialChirp = exponential_chirp_chain()
# OFDM data symbols
sound = map_to_transmit(K, CP, pilotValue, pilotCarriers, dataCarriers, bitsSP)
# sound = sound / np.max(sound)

# Known random OFDM block for channel estimation
knownOFDMBlock = known_ofdm_block(blockNum, seedStart, mu, K, CP, mappingTable)
# knownOFDMBlock = knownOFDMBlock / np.max(knownOFDMBlock)

# Total data sent over channel
# dataTotal = np.concatenate((np.zeros(fs), exponentialChirp.ravel(), (np.zeros(fs * time_before_data)), knownOFDMBlock, sound))

# If just using pilot tones, no known OFDM between data
dataTotal = np.concatenate((np.zeros(fs), exponentialChirp.ravel(), knownOFDMBlock, sound, np.zeros(fs)))

# If using known OFDM within data blocks
# dataTotal = np.concatenate((np.zeros(fs), exponentialChirp.ravel(), knownOFDMBlock))

# for i in range(numOFDMblocks//5):
#    dataTotal = np.concatenate((dataTotal, sound[int(i*(N+CP)):int((i+5)*(N+CP))], known_ofdm_block(1, seedStart, mu, K, CP, mappingTable)))

# print(len(dataTotal))
# dataTotal = np.concatenate((dataTotal, sound[230*(N+CP):235*(N+CP)], np.zeros(fs)))

# print(len(dataTotal))

# save(dataTotal, "audio/text_ldpc_1.wav")

plt.plot(dataTotal)
plt.title("Signal to send"); plt.xlabel('Sample number'); plt.ylabel('Sound amplitude');
plt.show()

#write("audio/exp_sounds/asyoulikeit_251_4_bandlimit35_125.wav", fs, dataTotal)

### CHANNEL ###

channelResponse = np.array(pd.read_csv("./channel/channel.csv", header = None)).ravel() 
noiseSNR = 100
ofdmReceived = channel(dataTotal, channelResponse, noiseSNR)
# ofdmReceived = np.append(np.zeros(10000), ofdmReceived)
HChannel = np.fft.fft(channelResponse, N)

### RECEIVER ###

# Channel Estimation Test

knownOFDMBlockReceived = channel(knownOFDMBlock, channelResponse, noiseSNR)
hestAtSymbols, _ = channel_estimate_known_ofdm(knownOFDMBlockReceived, seedStart, mappingTable, N, K, CP, mu, plot = False)

plt.plot(np.arange(N), HChannel, label = 'actual H')
plt.plot(np.arange(N), abs(hestAtSymbols), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

# Symbol Recovery Test
positionChirpEnd = chirp_synchroniser(ofdmReceived)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd 
ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum 
dataEnd = ofdmBlockEnd + numOFDMblocks * (N + CP)

# If want to use known OFDM in data
# dataEnd = ofdmBlockEnd + (numOFDMblocks + numOFDMblocks//5) * (N + CP) 

hest, offsetsss = channel_estimate_known_ofdm(ofdmReceived[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
plt.plot(np.arange(N), HChannel, label = 'actual H')
plt.plot(np.arange(N), abs(hest), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

print(offsetsss)
# plt.plot(ofdmReceived[ofdmBlockEnd:dataEnd])
# plt.show()

equalizedSymbols, hestAggregate = map_to_decode(ofdmReceived[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offset=0, samplingMismatch = 0, pilotImportance = 0, pilotValues = True, knownOFDMImportance = 0, knownOFDMInData = False)

# Noise variances shown for now
noiseVariances = [1, 1]
llrsReceived = return_llrs(equalizedSymbols, hestAggregate, noiseVariances)[:-numZerosAppend]
llrsReceived = np.reshape(llrsReceived[:len(llrsReceived)//648*648], (-1, 2 * ldpcCoder.K))
fullOutputData = []
for block in llrsReceived:
    outputData, _ = ldpcCoder.decode(block)
    np.place(outputData, outputData>0, int(0))
    np.place(outputData, outputData<0, int(1))
    fullOutputData.append(outputData[0:ldpcBlockLength])
outputData = np.array(fullOutputData).ravel()

#outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

#for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
#    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
#    plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
#plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
#plt.show()

dataToCsv = np.array(outputData, dtype=int)[:lenData]
demodulatedOutput = ''.join(str(e) for e in dataToCsv)
print(text_from_bits(demodulatedOutput))

ber = calculateBER(actualData, dataToCsv)
print("BER: " + str(ber))
