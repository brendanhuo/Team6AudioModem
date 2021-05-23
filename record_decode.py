from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from graphing_utils import *
from audio_utils import *

with open("./text/lorem.txt") as f:
    contents = f.read()

maximum_likelihood_estimation = False

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())

dataCarriers, pilotCarriers = assign_data_pilot(K, P)

receivedSound = np.load("audio/sound3.npy")
# receivedSound = audioDataFromFile("audio/TASCAM_0124.wav")
plt.plot(receivedSound)
plt.show()
# plot_waveform(receivedSound)

positionChirpEnd = chirp_synchroniser(receivedSound)
positionChirpEnd = round(positionChirpEnd)
print(positionChirpEnd)
# OFDM block channel estimation

if maximum_likelihood_estimation:

    detected_position = positionChirpEnd
    positionChirpEnd -= 100
    BER = []

    for i in range(201):

        positionChirpEnd += 1
        ofdmBlockStart = positionChirpEnd
        ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
        dataEnd = ofdmBlockEnd + 4 * (N + CP)  # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

        hest = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)

        equalizedSymbols = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance=0.49, pilotValues=True)
        outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

        dataToCsv = outputData.ravel()[0:len(ba)]
        demodulatedOutput = ''.join(str(e) for e in dataToCsv)

        ber = calculateBER(ba, dataToCsv)

        print("Bit Error Rate:" + str(ber))
        BER.append(ber)

    BER = np.array(BER)
    max_position = np.where(BER == np.amin(BER))[0][0] - 100 + detected_position

    plt.plot(np.arange(len(BER)) - (len(BER) + 1) / 2, BER)
    plt.title("Maximum BER occurs {} samples away from the detected point".format(detected_position - max_position))
    plt.xlabel("Sample number relative to detection point")
    plt.ylabel("BER")
    plt.show()

    print("Maximum BER occurs {} samples away from the detected point".format(detected_position - max_position))
    positionChirpEnd = max_position

    ofdmBlockStart = positionChirpEnd
    ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
    dataEnd = ofdmBlockEnd + 4 * (N + CP)  # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

    hest = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)

    equalizedSymbols = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance=0.49, pilotValues=True)
    outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

    dataToCsv = outputData.ravel()[0:len(ba)]
    demodulatedOutput = ''.join(str(e) for e in dataToCsv)

    print(text_from_bits(demodulatedOutput))
    ber = calculateBER(ba, dataToCsv)

    print("Bit Error Rate:" + str(ber))

else:
    ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
    dataEnd = ofdmBlockEnd + 4 * (N + CP) # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

    hest = channel_estimate_known_ofdm(receivedSound[positionChirpEnd: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
    plt.semilogy(np.arange(N), abs(hest), label='Estimated H via known OFDM')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.show()

    hestImpulse = np.fft.ifft(hest)[0:N//2]
    plt.plot(np.arange(300), hestImpulse[0:300])
    plt.title('Impulse response')
    plt.xlabel("Sample number")
    plt.ylabel("Impulse response magnitude")
    plt.show()

    equalizedSymbols = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance = 0.49, pilotValues = True)
    outputData, hardDecision = demapping(equalizedSymbols , demappingTable)
    z = np.arange(N//2-1)
    plt.scatter(equalizedSymbols[0:N//2-1].real, equalizedSymbols[0:N//2-1].imag, c=z, cmap="bwr")
    plt.show()
    #for qpsk, hard in zip(equalizedSymbols[2*(N//2-P-1):11000], hardDecision[2*(N//2-P-1):11000]):
    #    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    #    plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
    #plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
    plt.show()

    dataToCsv = outputData.ravel()[0:len(ba)]
    demodulatedOutput = ''.join(str(e) for e in dataToCsv)
    print(text_from_bits(demodulatedOutput))
    ber = calculateBER(ba, dataToCsv)

    print("Bit Error Rate:" + str(ber))
