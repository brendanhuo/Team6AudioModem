# Decode files that follow the SoundBridge343 standard

from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from audio_utils import *

# image_path = "./image/autumn_small.tif"
# ba, image_shape = image2bits(image_path, plot = True)
# imageData = ba
# file = "./text/asyoulik.txt"
# file = "./text/lorem.txt"
file = "./image/autumn_small.tif"
# # file = "audio/James/chirp length/lorem_2.0s.wav"
actualfileformat = file[-3:]

# Reads file depending on detected format
if actualfileformat == 'txt':
    with open(file) as f:
        contents = f.read()
        contents = contents[:len(contents)//2]
    ba = bitarray.bitarray()
    ba.frombytes(contents.encode('utf-8'))
    ba = np.array(ba.tolist())
    
elif actualfileformat == 'tif':
    # ba, _ = image2bits(file, plot=False)
    
    with open(file, 'rb') as f:
        info_bytes = f.read()
    bit_string = ''
    for i in info_bytes:
        binary_string = '{0:08b}'.format(i)
        bit_string += binary_string
    bit_int_list = [int(b) for b in str(bit_string)]
    ba = bit_int_list
    print(ba)
actualData = ba

useldpc = True
# LDPC encoding
if useldpc:
    # Pad ba for ldpc
    lenAppendldpc = ((len(ba)) // ldpcBlockLength + 1) * ldpcBlockLength - len(ba)
    ba = np.append(ba, np.random.binomial(n=1, p=0.5, size=(lenAppendldpc, )))
    ba = np.reshape(ba, (-1, ldpcBlockLength))

    ldpcConvert = []
    for i in range(len(ba)):
        encoded = ldpcCoder.encode(ba[i])
        ldpcConvert.append(encoded)

    ba = np.array(ldpcConvert).ravel()
lenData = len(ba)

dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = useBandLimit)

sync_error = []
sampling_mismatches = []
bers = []

for i in range(1):
    # MAKE SURE THAT THE INPUT HAS GLOBAL VALUES THAT MATCH
    # receivedSound = audioDataFromFile("audio/brendan/16qam/outside/TASCAM_0{}.wav".format(i+300))
    receivedSound = audioDataFromFile("audio/brendan/autumn/large/QAM/TASCAM_0311.wav")   
    # plt.plot(np.arange(len(receivedSound))/fs, receivedSound)
    # plt.title('Received Sound'); plt.xlabel('Time/s'); plt.ylabel('Sound amplitude')
    # plt.show()

    # Symbol Recovery Test
    positionChirpEnd = chirp_synchroniser(receivedSound)

    # OFDM block channel estimation
    ofdmBlockStart = positionChirpEnd + (N + CP) * preblocknum
    ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum + (N + CP) * preblocknum
    dataStart = ofdmBlockEnd + (N + CP) * postblocknum

    hest, offset = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu, plot = False)
    print('Synchronization offset is: ' + str(offset))
    sync_error.append(offset)
    # Correct for synchronization error
    ofdmBlockStart = positionChirpEnd + floor(offset) + (N + CP) * preblocknum
    ofdmBlockEnd = positionChirpEnd  + (N + CP) * blockNum + floor(offset) + (N + CP) * preblocknum
    dataStart = ofdmBlockEnd + (N + CP) * postblocknum

    # Remaining offset to rotate
    offset = offset - floor(offset)

    # Plot new channel estimates
    hest, offset = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu, plot = False)
    print('New synchronization offset (for rotation): ' + str(offset))

    ### EXTRACT METADATA ###
    # lenData, numOFDMblocks, file_format = extract_Metadata(dataCarriers, receivedSound, dataStart, hest, pilotCarriers)
    numOFDMblocks = lenData//mu//len(dataCarriers) + 1

    # print("estimated length: ", lenData + len_metadata_bits)
    dataEnd = dataStart + (numOFDMblocks + numOFDMblocks//knownInDataFreq + 1) * (N + CP) 
    lenAppendldpc = ((lenData) // ldpcBlockLength + 1) * ldpcBlockLength - lenData
    lenTotalba = lenData + lenAppendldpc
    numZerosAppend = len(dataCarriers)*mu - (lenTotalba - lenTotalba//mu//len(dataCarriers) * len(dataCarriers) * mu)

    # Calculate sampling mismatch from difference in sampling frequency between speakers and microphone
    sampleMismatch = calculate_sampling_mismatch(receivedSound[dataStart:dataEnd], hest, N, CP, pilotCarriers, pilotValue, offset = offset, plot = False)
    print('Sampling mismatch is: ' + str(sampleMismatch))
    sampling_mismatches.append(sampleMismatch)
    equalizedSymbols, hestAggregate = map_to_decode(receivedSound[dataStart:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offset=offset, samplingMismatch=sampleMismatch, pilotImportance=pilotImportance,
                                                        pilotValues=pilotValues, knownOFDMImportance=knownOFDMImportance, knownOFDMInData=knownOFDMInData, plot = True)

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
    lenData = len(actualData)
    file_format = 'tif'
    dataToCsv = np.array(outputData, dtype=int).ravel()[len_metadata_bits:len_metadata_bits + lenData]
    # file_format = 'txt'
    if file_format == 'txt':
        demodulatedOutput = ''.join(str(e) for e in dataToCsv)
        print(text_from_bits(demodulatedOutput))

    elif file_format == 'tif':
        string_ints = [str(it) for it in dataToCsv]
        string_of_bits = "".join(string_ints)
        image_bits = bitarray(string_of_bits)
        bits_image_string = str(image_bits)
        image_bytes = image_bits.tobytes()
        with open('result/data_output.tif', 'wb') as f:
            f.write(image_bytes)

    elif file_format == 'wav':
        save(dataToCsv, "audio/James/Decoded Outputs/output.wav{}".format(file_format))
        print(len(dataToCsv)/fs)
        play(dataToCsv)

    ber = calculateBER(actualData, dataToCsv)
    print("Bit Error Rate:" + str(ber))
    bers.append(ber)

# np.save("bers_16qam_outside.npy",np.asarray(bers))
# np.save("syncerror_16qam_outside.npy",np.asarray(sync_error))
# np.save("samplemismatch_16qam_outside.npy",np.asarray(sampling_mismatches))