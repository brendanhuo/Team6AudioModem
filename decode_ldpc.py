from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 
from ldpc_jossy.py import ldpc

# Import text file for testing
with open("./text/asyoulik.txt") as f:
    contents = f.read()
    contents = contents[:len(contents)//4] # Make sure this is actually the data you sent 

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())

actualData = ba
lenData = len(ba)
print(lenData)
dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = True) # Make sure to check if you actually sent it as bandlimited

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

# Pad ba to fill full OFDM blocks - use random data 
numZerosAppend = len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)
ba = np.append(ba, np.zeros(numZerosAppend))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
numOFDMblocks = len(bitsSP)

# MAKE SURE THAT THE INPUT HAS GLOBAL VALUES THAT MATCH
receivedSound = np.load("audio/brendan/asyoulikeit-rec4_125_bandlimit35.npy")
plt.plot(np.arange(len(receivedSound))/fs, receivedSound)
plt.title('Received Sound'); plt.xlabel('Time/s'); plt.ylabel('Sound amplitude')
plt.show()

# Synchronization using chirp
positionChirpEnd = chirp_synchroniser(receivedSound)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd
ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
dataEnd = ofdmBlockEnd + numOFDMblocks * (N + CP) # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

hest, offset = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
print('Synchronization offset is: ' + str(offset))

plt.semilogy(np.arange(N)*fs / N, abs(hest))
plt.grid(True); plt.title('Estimated H via known OFDM'); plt.xlabel('Frequency/Hz'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

hestImpulse = np.fft.ifft(hest)[0:N//2]
plt.plot(np.arange(N//2 )/fs, hestImpulse[0:N//2])
plt.title('Impulse response'); plt.xlabel('Time / s'); plt.ylabel('Amplitude of impulse response')
plt.show()

# Correct for synchronization error
ofdmBlockStart = positionChirpEnd + floor(offset) 
ofdmBlockEnd = positionChirpEnd  + (N + CP) * blockNum + floor(offset) 
dataEnd = ofdmBlockEnd + numOFDMblocks * (N + CP) # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

# Remaining offset to rotate
offset = offset - floor(offset)

# Plot new channel estimates
hest, offset = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
print('New synchronization offset (for rotation): ' + str(offset))

plt.semilogy(np.arange(N)*fs / N, abs(hest))
plt.grid(True); plt.title('Estimated H via known OFDM'); plt.xlabel('Frequency/Hz'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

hestImpulse = np.fft.ifft(hest)[0:N//2]
plt.plot(np.arange(N//2)/fs, hestImpulse[0:N//2])
plt.title('Impulse response'); plt.xlabel('Time / s'); plt.ylabel('Amplitude of impulse response')
plt.show()

# Calculate sampling mismatch from difference in sampling frequency between speakers and microphone
sampleMismatch = calculate_sampling_mismatch(receivedSound[ofdmBlockEnd:dataEnd], hest, N, CP, pilotCarriers, pilotValue, offset = offset, plot = True)

# Calculate equalized symbols using pilot tone assisted channel frequency response estimate
equalizedSymbols, hestAggregate = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offset = offset, samplingMismatch = sampleMismatch, pilotImportance = 0.5, pilotValues = True)

# Decode using LDPC
# Noise variances that are estimated - real part and imaginary part - may want to refine later
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

# Bit sequence to text
dataToCsv = np.array(outputData, dtype=int)[:lenData]
demodulatedOutput = ''.join(str(e) for e in dataToCsv)
print(text_from_bits(demodulatedOutput))

# Calculate bit error rate (BER)
ber = calculateBER(actualData, dataToCsv)
print("Bit Error Rate:" + str(ber))
