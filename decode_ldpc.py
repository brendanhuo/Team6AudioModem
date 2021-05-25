from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 
from ldpc_jossy.py import ldpc

# Import text file for testing
with open("./text/lorem.txt") as f:
    contents = f.read()

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())

dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = False)

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

numZerosAppend = len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)
ba = np.append(ba, np.zeros(numZerosAppend))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
numOFDMblocks = len(bitsSP)

receivedSound = np.load("audio/text_ldpc_received_1.npy")
plt.plot(receivedSound)
plt.show()

positionChirpEnd = chirp_synchroniser(receivedSound)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd
ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
dataEnd = ofdmBlockEnd + numOFDMblocks * (N + CP) # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

hest = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
plt.semilogy(np.arange(N), abs(hest), label='Estimated H via known OFDM')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

hestImpulse = np.fft.ifft(hest)[0:N//2]
plt.plot(np.arange(N//2), hestImpulse[0:N//2])
plt.title('Impulse response')
plt.show()

# Decode using LDPC

equalizedSymbols, hestAggregate = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance = 0.5, pilotValues = True)

# Noise variances that are estimated - real part and imaginary part - may want to refine later
noiseVariances = [1, 1]
llrsReceived = return_llrs(equalizedSymbols, hestAggregate, noiseVariances)[:-numZerosAppend]
llrsReceived = np.reshape(llrsReceived, (-1, 2 * ldpcCoder.K))
outputData = []
for block in llrsReceived:
    ldpcDecode, _ = ldpcCoder.decode(block)
    outputData.append(ldpcDecode)
outputData = np.array(outputData).ravel()
np.place(outputData, outputData>0, int(0))
np.place(outputData, outputData<0, int(1))

dataToCsv = np.array(outputData, dtype=int)
demodulatedOutput = ''.join(str(e) for e in dataToCsv)
print(text_from_bits(demodulatedOutput))
ber = calculateBER(ba, dataToCsv)
print("Bit Error Rate:" + str(ber))
