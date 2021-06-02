from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 
from ldpc_jossy.py import ldpc

dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited=True)

### TRANSMITTER ###
image_path = "./image/autumn_small.tif"
ba, image_shape = image2bits(image_path, plot = True)
imageData = ba
lenData = len(ba)
# print(lenData)
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
# print(len(ba))
dataToCsv = ba.astype(int).ravel()[:lenData]

byte_array = []
for i in range (len(dataToCsv)//8):
    demodulatedOutput = ''.join(str(e) for e in dataToCsv[8*i:8*(i+1)])
    byte_array.append(int(demodulatedOutput,2))
lenBytes = 1
for shape in image_shape:
    lenBytes *= shape
plt.imshow(np.array(byte_array)[0:lenBytes].reshape(image_shape))
plt.show()

#pad ba for OFDM block length
numZerosAppend = len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)
#ba = np.append(ba, np.zeros(len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)))
ba = np.append(ba, np.random.binomial(n=1, p=0.5, size=(numZerosAppend, )))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
numOFDMblocks = len(bitsSP)
# print(numOFDMblocks)

# print(len(ba))
# Chirp 
exponentialChirp = exponential_chirp()

# OFDM data symbols
sound = map_to_transmit(K, CP, pilotValue, pilotCarriers, dataCarriers, bitsSP)

# Known random OFDM block for channel estimation
knownOFDMBlock = known_ofdm_block(blockNum, seedStart, mu, K, CP, mappingTable)

# Total data sent over channel
dataTotal = np.concatenate((np.zeros(44100), exponentialChirp.ravel(), knownOFDMBlock, sound, np.zeros(fs)))

plt.plot(dataTotal)
plt.title("Signal to send")
plt.xlabel('Sample number');plt.ylabel('Sound amplitude');
plt.show()

write("audio/brendan/autumn-test_test1.wav", fs, dataTotal)

# ### CHANNEL ###

# channelResponse = np.array(pd.read_csv("./channel/channel.csv", header = None)).ravel() 
# noiseSNR = 50
# ofdmReceived = channel(dataTotal, channelResponse, noiseSNR)
# # ofdmReceived = np.append(np.zeros(10000), ofdmReceived)
# HChannel = np.fft.fft(channelResponse, N)
# # print(len(ofdmReceived))

# ### RECEIVER ###

# # Channel Estimation Test

# knownOFDMBlockReceived = channel(knownOFDMBlock, channelResponse, noiseSNR)
# hestAtSymbols, _ = channel_estimate_known_ofdm(knownOFDMBlockReceived, seedStart, mappingTable, N, K, CP, mu)

# plt.semilogy(np.arange(N), HChannel, label = 'actual H')
# plt.semilogy(np.arange(N), abs(hestAtSymbols), label='Estimated H')
# plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
# plt.show()

# # Symbol Recovery Test
# positionChirpEnd = chirp_synchroniser(ofdmReceived)

# # OFDM block channel estimation
# ofdmBlockStart = positionChirpEnd
# ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
# dataEnd = ofdmBlockEnd + numOFDMblocks * (N + CP) 

# hest, _ = channel_estimate_known_ofdm(ofdmReceived[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
# plt.semilogy(np.arange(N), HChannel, label = 'actual H')
# plt.semilogy(np.arange(N), abs(hest), label='Estimated H')
# plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
# plt.show()

# plt.plot(ofdmReceived[ofdmBlockEnd:])
# plt.show()

# equalizedSymbols, hestAggregate = map_to_decode(ofdmReceived[ofdmBlockEnd:], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offsets = [0,0,0], samplingMismatch = 0, pilotImportance = 0.5, pilotValues=True)
# outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

# print(len(equalizedSymbols))
# # Noise variances shown for now
# noiseVariances = [0.01, 0.01]
# llrsReceived = return_llrs(equalizedSymbols, hestAggregate, noiseVariances)[:-numZerosAppend]
# llrsReceived = np.reshape(llrsReceived, (-1, 2 * ldpcCoder.K))
# outputData = []
# for block in llrsReceived[:len(llrsReceived)]:
#     ldpcDecode, _ = ldpcCoder.decode(block, 'sumprod2')
#     np.place(ldpcDecode, ldpcDecode>0, int(0))
#     np.place(ldpcDecode, ldpcDecode<0, int(1))
#     outputData.append(ldpcDecode[0:324])
# outputData = np.array(outputData).ravel()

# #for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
# #    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
# #    #plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
# #plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
# #plt.show()

# #np.save("image_decode_test.npy", np.asarray(outputData))
# dataToCsv = outputData.astype(int).ravel()[:lenData]
# ber = calculateBER(imageData, dataToCsv)
# print("Bit Error Rate:" + str(ber))
# byte_array = []
# for i in range (len(dataToCsv)//8):
#     demodulatedOutput = ''.join(str(e) for e in dataToCsv[8*i:8*(i+1)])
#     byte_array.append(int(demodulatedOutput,2))
# lenBytes = 1
# for shape in image_shape:
#     lenBytes *= shape
# plt.imshow(np.array(byte_array)[0:lenBytes].reshape(image_shape))
# plt.show()



