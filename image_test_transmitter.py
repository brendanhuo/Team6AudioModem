from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 

dataCarriers, pilotCarriers = assign_data_pilot(K, P)

### TRANSMITTER ###
image_path = "./image/bali.tif"
ba = image2bits(image_path)
print(len(ba))

ba = np.append(ba, np.zeros(len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
print(len(bitsSP))

# Chirp 
exponentialChirp = exponential_chirp()

# OFDM data symbols
sound = map_to_transmit(K, CP, pilotValue, pilotCarriers, dataCarriers, bitsSP)

# Known random OFDM block for channel estimation
knownOFDMBlock = known_ofdm_block(blockNum, seedStart, mu, K, CP, mappingTable)

# Total data sent over channel
dataTotal = np.concatenate((np.zeros(44100), exponentialChirp.ravel(), knownOFDMBlock, sound))

plt.plot(dataTotal)
plt.title("Signal to send")
plt.show()

write("audio/img_sound_send_1.wav", fs, dataTotal)

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

plt.semilogy(np.arange(N), HChannel, label = 'actual H')
plt.semilogy(np.arange(N), abs(hestAtSymbols), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

# Symbol Recovery Test
positionChirpStart = chirp_synchroniser(ofdmReceived)

# OFDM block channel estimation
ofdmBlockStart = positionChirpStart + chirp_length * fs
ofdmBlockEnd = positionChirpStart + chirp_length * fs + (N + CP) * blockNum
dataEnd = ofdmBlockEnd + 1584 * (N + CP) # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

hest = channel_estimate_known_ofdm(ofdmReceived[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
plt.semilogy(np.arange(N), HChannel, label = 'actual H')
plt.semilogy(np.arange(N), abs(hest), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

plt.plot(ofdmReceived[ofdmBlockEnd:])
plt.show()

equalizedSymbols = map_to_decode(ofdmReceived[ofdmBlockEnd:], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance, pilotValues)
outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    #plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
plt.show()

dataToCsv = outputData.ravel()[:len(ba)]
byte_array = []
for i in range (len(dataToCsv)//8):
    demodulatedOutput = ''.join(str(e) for e in dataToCsv[8*i:8*(i+1)])
    byte_array.append(int(demodulatedOutput,2))
plt.imshow(np.array(byte_array)[0:1418100].reshape(489, 725, 4))
plt.show()



