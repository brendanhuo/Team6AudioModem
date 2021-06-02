# Decode files that follow the SoundBridge343 standard

from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from audio_utils import *

image_path = "./image/autumn_small.tif"
ba, image_shape = image2bits(image_path, plot = True)
imageData = ba

useldpc = True
dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = useBandLimit)

# MAKE SURE THAT THE INPUT HAS GLOBAL VALUES THAT MATCH
receivedSound = audioDataFromFile("audio/brendan/testing/autumn_standard.wav")
plt.plot(np.arange(len(receivedSound))/fs, receivedSound)
plt.title('Received Sound'); plt.xlabel('Time/s'); plt.ylabel('Sound amplitude')
plt.show()

# Symbol Recovery Test
positionChirpEnd = chirp_synchroniser(receivedSound)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd + (N + CP) * preblocknum
ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum + (N + CP) * preblocknum
dataStart = ofdmBlockEnd + (N + CP) * postblocknum

hest, offset = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu, plot = True)
print('Synchronization offset is: ' + str(offset))

# Correct for synchronization error
ofdmBlockStart = positionChirpEnd + floor(offset) + (N + CP) * preblocknum
ofdmBlockEnd = positionChirpEnd  + (N + CP) * blockNum + floor(offset) + (N + CP) * preblocknum
dataStart = ofdmBlockEnd + (N + CP) * postblocknum

# Remaining offset to rotate
offset = offset - floor(offset)

# Plot new channel estimates
hest, offset = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu, plot = True)
print('New synchronization offset (for rotation): ' + str(offset))

### EXTRACT METADATA ###
lenData, numOFDMblocks, file_format = extract_Metadata(dataCarriers, receivedSound, dataStart, hest, pilotCarriers)

# print("estimated length: ", lenData + len_metadata_bits)
dataEnd = dataStart + (numOFDMblocks + numOFDMblocks//knownInDataFreq) * (N + CP) 
lenAppendldpc = ((lenData) // ldpcBlockLength + 1) * ldpcBlockLength - lenData
lenTotalba = lenData + lenAppendldpc
numZerosAppend = len(dataCarriers)*mu - (lenTotalba - lenTotalba//mu//len(dataCarriers) * len(dataCarriers) * mu)

# Calculate sampling mismatch from difference in sampling frequency between speakers and microphone
sampleMismatch = calculate_sampling_mismatch(receivedSound[dataStart:dataEnd], hest, N, CP, pilotCarriers, pilotValue, offset = offset, plot = True)
print('Sampling mismatch is: ' + str(sampleMismatch))

equalizedSymbols, hestAggregate = map_to_decode(receivedSound[dataStart:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offset=offset, samplingMismatch=sampleMismatch, pilotImportance=pilotImportance,
                                                    pilotValues=pilotValues, knownOFDMImportance=knownOFDMImportance, knownOFDMInData=knownOFDMInData)

# Noise variances shown for now
noiseVariances = [1, 1]
if useldpc:
    outputData = ldpcDecode(equalizedSymbols, hestAggregate, noiseVariances, numZerosAppend)
else:
    outputData, hardDecision = demapping(equalizedSymbols, demappingTable)
    for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
        plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
        plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
        plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
    plt.show()

dataToCsv = np.array(outputData, dtype=int).ravel()[len_metadata_bits:len_metadata_bits + lenData]

if file_format == 1:
    demodulatedOutput = ''.join(str(e) for e in dataToCsv)
    print(text_from_bits(demodulatedOutput))

elif file_format == 2:
    byte_array = []
    for i in range(len(dataToCsv) // 8):
        demodulatedOutput = ''.join(str(e) for e in dataToCsv[8 * i:8 * (i + 1)])
        byte_array.append(int(demodulatedOutput, 2))
    lenBytes = 1
    for shape in image_shape:
        lenBytes *= shape
    plt.imshow(np.array(byte_array)[0:lenBytes].reshape(image_shape))
    plt.title('Image received')
    plt.show()

elif file_format == 3:
    save(dataToCsv, "audio/James/Decoded Outputs/output.wav{}".format(file_format))
    print(len(dataToCsv)/fs)
    play(dataToCsv)

ber = calculateBER(imageData, dataToCsv)
print("Bit Error Rate:" + str(ber))

