from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 
from ldpc_jossy.py import ldpc
from audio_utils import *

# Import text file for testing
with open("./text/asyoulik.txt") as f:
    contents = f.read()
    contents = contents[:len(contents)//2] # Make sure this is actually the data you sent 

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())

actualData = ba
lenData = len(ba)
print(lenData)
dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = useBandLimit) # Make sure to check if you actually sent it as bandlimited

# LDPC encoding

ldpcCoder = ldpc.code()
ldpcBlockLength = ldpcCoder.K

# Pad ba for ldpc
lenAppendldpc = ((len(ba)) // ldpcBlockLength + 1) * ldpcBlockLength - len(ba)
ba = np.append(ba, np.random.binomial(n=1, p=0.5, size=(lenAppendldpc, )))
ldpcConvert = []
for i in range(len(ba)):
    encoded = ldpcCoder.encode(ba[i])
    ldpcConvert.append(encoded)
ba = np.array(ldpcConvert).ravel()

# Pad ba to fill full OFDM blocks - use random data 
numZerosAppend = len(dataCarriers)*mu - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)
ba = np.append(ba, np.zeros(numZerosAppend))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
numOFDMblocks = len(bitsSP)

# MAKE SURE THAT THE INPUT HAS GLOBAL VALUES THAT MATCH
receivedSound = audioDataFromFile("audio/brendan/clean_sound/TASCAM_0199.wav")
plt.plot(np.arange(len(receivedSound))/fs, receivedSound)
plt.title('Received Sound'); plt.xlabel('Time/s'); plt.ylabel('Sound amplitude')
plt.show()

# Synchronization using chirp
positionChirpEnd = chirp_synchroniser(receivedSound)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd + (N + CP) 
ofdmBlockEnd = positionChirpEnd  + (N + CP) * blockNum 
dataStart = ofdmBlockEnd
dataEnd = ofdmBlockEnd + numOFDMblocks * (N + CP) # number of data OFDM blocks we are sending should be determined by metadata

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
dataStart = ofdmBlockEnd
dataEnd = ofdmBlockEnd + numOFDMblocks * (N + CP) # number of data OFDM blocks we are sending should be determined by metadata

# Remaining offset to rotate
offset = offset - floor(offset)

# Plot new channel estimates
hest, offset = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
print('New synchronization offset (for rotation): ' + str(offset))

plt.semilogy(dataCarriers*fs / N, abs(hest[dataCarriers]))
plt.grid(True); plt.title('Estimated H via known OFDM'); plt.xlabel('Frequency/Hz'); plt.ylabel('$|H(f)|$'); 
plt.show()

hestImpulse = np.fft.ifft(hest)[0:N//2]
plt.plot(np.arange(N//2)/fs, hestImpulse[0:N//2])
plt.title('Impulse response'); plt.xlabel('Time / s'); plt.ylabel('Amplitude of impulse response')
plt.show()

# Calculate sampling mismatch from difference in sampling frequency between speakers and microphone
sampleMismatch = calculate_sampling_mismatch(receivedSound[dataStart:dataEnd], hest, N, CP, pilotCarriers, pilotValue, offset = offset, plot = True)
print('Sampling mismatch is: ' + str(sampleMismatch))

# Calculate equalized symbols using pilot tone assisted channel frequency response estimate
equalizedSymbols, hestAggregate = map_to_decode(receivedSound[dataStart:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offset = offset, samplingMismatch = sampleMismatch, pilotImportance = 0.5, pilotValues = True)

# outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

# for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
#    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
#    plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
# plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
# plt.show()

# Decode using LDPC
# Noise variances that are estimated - real part and imaginary part - may want to refine later
noiseVariances = [1, 1]
outputData = ldpcDecode(equalizedSymbols, hestAggregate, noiseVariances, numZerosAppend)

# Bit sequence to text
dataToCsv = np.array(outputData, dtype=int).ravel()[:lenData]
print(dataToCsv)
demodulatedOutput = ''.join(str(e) for e in dataToCsv)
print(demodulatedOutput)
print(text_from_bits(demodulatedOutput))

# Calculate bit error rate (BER)
ber = calculateBER(actualData, dataToCsv)
print("Bit Error Rate:" + str(ber))

np.save('text_decoded6.npy', dataToCsv)
