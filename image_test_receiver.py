from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 
from ldpc_jossy.py import ldpc
from statistics import mean 
from audio_utils import *

image_path = "./image/autumn_small.tif"
ba, image_shape = image2bits(image_path, plot = True)
imageData = ba
lenData = len(ba)
print("Length of data:" + str(lenData))

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
print("Length in bits after LDPC:" + str(len(ba)))

dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = True)

#pad ba for OFDM block length
numZerosAppend = len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)
#ba = np.append(ba, np.zeros(len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)))
ba = np.append(ba, np.random.binomial(n=1, p=0.5, size=(numZerosAppend, )))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
numOFDMblocks = len(bitsSP)
print("Number of OFDM blocks: " + str(numOFDMblocks))

ba = np.append(ba, np.zeros(len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
numOFDMblocks = len(bitsSP)

receivedSound = audioDataFromFile("audio/brendan/TASCAM_0222.wav") #6_ldpc is 4096/256/251
plt.plot(receivedSound)
plt.show()

# Synchronization using chirp
positionChirpEnd = chirp_synchroniser(receivedSound)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd + (N + CP) * noiseBlocks
ofdmBlockEnd = positionChirpEnd  + (N + CP) * blockNum + (N + CP) * noiseBlocks
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
ofdmBlockStart = positionChirpEnd + floor(offset) + (N + CP) * noiseBlocks
ofdmBlockEnd = positionChirpEnd  + (N + CP) * blockNum + floor(offset) + (N + CP) * noiseBlocks
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
llrsReceived = return_llrs(equalizedSymbols, hestAggregate, noiseVariances)[:-numZerosAppend]
llrsReceived = np.reshape(llrsReceived[:len(llrsReceived)//648*648], (-1, 2 * ldpcCoder.K))
fullOutputData = []
for block in llrsReceived:
    outputData, _ = ldpcCoder.decode(block, 'sumprod2')
    for i in range(len(outputData[0:ldpcBlockLength])):
        if outputData[i] > 0:
            outputData[i] = int(0)
        else:
            outputData[i] = int(1)
    fullOutputData.append(outputData[0:ldpcBlockLength])    
    # np.place(outputData, outputData>=0, int(0))
    # np.place(outputData, outputData<0, int(1))
    # fullOutputData.append(outputData[0:ldpcBlockLength])
outputData = np.array(fullOutputData)

'''
z = np.arange(N//2-1)
plt.scatter(equalizedSymbols[0:N//2-1].real, equalizedSymbols[0:N//2-1].imag, c=z, cmap="bwr")
plt.show()
for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
plt.show()'''

dataToCsv = outputData.astype(int).ravel()[:lenData]

ber = calculateBER(imageData, dataToCsv)
print("Bit Error Rate:" + str(ber))
'''
def returnError(data, ba, i):
    errors = np.zeros((N//2-P-1)*2)
    for l in range((N//2-P-1)*2):
        if data[l+i*(N//2-P-1)*2] != ba[l+i*(N//2-P-1)*2]:
            errors[l] += 1
    return errors
plt.figure(1)
plt.plot(returnError(dataToCsv, ba, 0))
plt.figure(2)
plt.plot(returnError(dataToCsv, ba, 10))
plt.figure(3)
plt.plot(returnError(dataToCsv, ba, 100))
plt.show()'''
byte_array = []
for i in range (len(dataToCsv)//8):
    demodulatedOutput = ''.join(str(e) for e in dataToCsv[8*i:8*(i+1)])
    byte_array.append(int(demodulatedOutput,2))
lenBytes = 1
for shape in image_shape:
    lenBytes *= shape
plt.imshow(np.array(byte_array)[0:lenBytes].reshape(image_shape))
plt.title('Image received')
plt.show()


        


