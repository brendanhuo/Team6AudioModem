from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from audio_utils import *
from ldpc_jossy.py import ldpc

useldpc = True
dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = useBandLimit)

### TRANSMITTER ###

# Import text file for testing

# file = "./text/asyoulik.txt"
# file = "./text/lorem.txt"
file = "./image/autumn_small.tif"
# file = "audio/James/chirp length/lorem_2.0s.wav"
actualfileformat = file[-3:]

# Reads file depending on detected format
if file_formats[actualfileformat] == 1:
    with open(file) as f:
        contents = f.read()
        # contents = contents[:len(contents)//8]
    ba = bitarray.bitarray()
    ba.frombytes(contents.encode('utf-8'))
    ba = np.array(ba.tolist())

elif file_formats[actualfileformat] == 2:
    ba, _ = image2bits(file, plot=False)

elif file_formats[actualfileformat] == 3:
    binary = []
    with open(file, "rb") as f:
        ba = f.read()
    for byte in ba:
        binary.append([int(i) for i in str(bin(byte)[2:].zfill(8))])
    ba = np.array(binary)
    ba = ba.reshape(-1)
    print(len(ba))
else:
    raise ValueError("Not a recognisable file format: formats include: ", file_formats)

actualData = ba
lenData0 = len(ba)

### Metadata Encoding ###
ba = append_Metadata(ba, file, lenData0)

# LDPC encoding
if useldpc:
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
numZerosAppend = len(dataCarriers)*mu - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)
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
knownOFDMBlock = known_ofdm_block(blockNum+noiseBlocks, seedStart-noiseBlocks, mu, K, CP, mappingTable)
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

write("audio/Brendan/testing/input_{}.wav".format(actualfileformat), fs, dataTotal)

### CHANNEL ###

channelResponse = np.array(pd.read_csv("./channel/channel.csv", header = None)).ravel() 
noiseSNR = 100
ofdmReceived = channel(dataTotal, channelResponse, noiseSNR)
# ofdmReceived = np.append(np.zeros(10000), ofdmReceived)
HChannel = np.fft.fft(channelResponse, N)

### RECEIVER ###

# Channel Estimation Test

knownOFDMBlockReceived = channel(knownOFDMBlock, channelResponse, noiseSNR)
hestAtSymbols, _ = channel_estimate_known_ofdm(knownOFDMBlockReceived, seedStart-noiseBlocks, mappingTable, N, K, CP, mu, plot = False)

plt.plot(np.arange(N), HChannel, label = 'actual H')
plt.plot(np.arange(N), abs(hestAtSymbols), label='Estimated H')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

# Symbol Recovery Test
positionChirpEnd = chirp_synchroniser(ofdmReceived)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd + (N + CP) * noiseBlocks
ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum + (N + CP) * noiseBlocks
dataStart = ofdmBlockEnd 
#dataEnd = dataStart + numOFDMblocks * (N + CP)

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


### EXTRACT METADATA ###

lenData, numOFDMblocks, file_format = extract_Metadata(dataCarriers, ofdmReceived, dataStart, hest, pilotCarriers, numZerosAppend, ldpcCoder, ldpcBlockLength)

# print("estimated length: ", lenData + len_metadata_bits)

equalizedSymbols, hestAggregate = map_to_decode(ofdmReceived[dataStart:dataStart + numOFDMblocks * (N + CP)], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offset=0, samplingMismatch=0, pilotImportance=0,
                                                    pilotValues=True, knownOFDMImportance=0, knownOFDMInData=False)

# Noise variances shown for now
noiseVariances = [1, 1]
def ldpcDecode(equalizedSymbols, hestAggregate, noiseVariances, numZerosAppend):
    if QPSK:
        llrsReceived = return_llrs(equalizedSymbols, hestAggregate, noiseVariances)
    elif QAM:
        llrsReceived = return_llrs_16qam(equalizedSymbols, hestAggregate, noiseVariances)[:-numZerosAppend]
    llrsReceived = np.reshape(llrsReceived[:len(llrsReceived)//648*648], (-1, 2 * ldpcCoder.K))
    fullOutputData = []
    i = 0
    for block in llrsReceived:
        percentage_0 = i * 100 // len(llrsReceived)
        i += 1
        percentage = i * 100 // len(llrsReceived)

        if percentage != percentage_0:
            print("{}% completed".format(percentage))
        outputData, _ = ldpcCoder.decode(block)
        np.place(outputData, outputData>0, int(0))
        np.place(outputData, outputData<0, int(1))
        fullOutputData.append(outputData[0:ldpcBlockLength])
    outputData = np.array(fullOutputData).ravel()
    return outputData
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

# print("lengths: ", lenData, len(dataToCsv), len(actualData))
ber = calculateBER(actualData, dataToCsv)
print("BER: " + str(ber))

# Get BER graph for LDPC

# SNRs = [1,2,5,10,15,20,25,40,50,100]
# bers_ldpc = []
# bers_noldpc = []
# for snr in SNRs:
#     print(snr)
#     ofdmReceived = channel(dataTotal, channelResponse, snr) 
#     hest, offset = channel_estimate_known_ofdm(ofdmReceived[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
#     equalizedSymbols, hestAggregate = map_to_decode(ofdmReceived[dataStart:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offset=0, samplingMismatch = 0, pilotImportance = 0, pilotValues = True, knownOFDMImportance = 0, knownOFDMInData = False, plot = False)
    
    # for when using LDPC
    # outputData = ldpcDecode(equalizedSymbols, hestAggregate, noiseVariances, numZerosAppend)
    
    # no LDPC
    # outputData, hardDecision = demapping(equalizedSymbols, demappingTable)
    # dataToCsv = np.array(outputData, dtype=int).ravel()[:lenData]
    # ber = calculateBER(actualData, dataToCsv)
    # bers_ldpc.append(ber)
    # bers_noldpc.append(ber)

# plt.plot(SNRs, bers_ldpc)
# plt.title('BER for different SNR using LDPC'); plt.xlabel('SNR'); plt.ylabel('BER');
# plt.show()
# np.save('bers_ldpc', np.asarray(bers_ldpc))

# bers_ldpc = np.load('bers_ldpc.npy')
# plt.plot(SNRs, bers_ldpc, label = 'With LDPC')
# plt.plot(SNRs, bers_noldpc, label = 'Without LDPC')
# plt.legend(); plt.title('BER for different SNR using LDPC'); plt.xlabel('SNR/dB'); plt.ylabel('BER');
# plt.show()
