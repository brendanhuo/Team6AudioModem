from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from graphing_utils import *
from audio_utils import *
from chirp_channel_estimation import *
from scipy import fft, ifft

def wav_transmission(array, filename, plot=True):
    """Takes inputted array and saves modulated wav file for playback"""

    ba = array
    dataCarriers, pilotCarriers = assign_data_pilot(K, P)
    ba = np.append(ba, np.zeros(len(dataCarriers) * 2 - (len(ba) - len(ba) // mu // len(dataCarriers) * len(dataCarriers) * mu)))
    bitsSP = ba.reshape((len(ba) // mu // len(dataCarriers), len(dataCarriers), mu))
    print(len(bitsSP))

    # Chirp
    exponentialChirp = exponential_chirp_chain()

    # OFDM data symbols
    sound = map_to_transmit(K, CP, pilotValue, pilotCarriers, dataCarriers, bitsSP)

    # Known random OFDM block for channel estimation
    knownOFDMBlock = known_ofdm_block(blockNum, seedStart, mu, K, CP, mappingTable)

    # Total data sent over channel
    dataTotal = np.concatenate((np.zeros(44100), exponentialChirp.ravel(), knownOFDMBlock, sound))

    if plot:
        plt.plot(dataTotal)
        plt.title("Signal to send")
        plt.show()

    write(filename, fs, dataTotal)


def decode_and_compare_text(y, x, plot=True, maximum_likelihood_estimation = True):
    """Decodes array and returns decoded file and BER"""

    positionChirpEnd = chirp_synchroniser(y)
    positionChirpEnd = round(positionChirpEnd)
    positionChirpEnd -= 0
    receivedSound = y
    ba = x
    print(len(ba))
    dataCarriers, pilotCarriers = assign_data_pilot(K, P)

    hest_chirp = Hest_from_chirp(y, plot=False)
    hest_chirp = hest_chirp / np.linalg.norm(hest_chirp)
    
    if plot:
        plot_frequency_response(hest_chirp)
        test = ifft(hest_chirp)
        plot_waveform(test)

    # OFDM block channel estimation

    if maximum_likelihood_estimation:

        estimated_bits = [0] * len(x)
        detected_position = positionChirpEnd
        bias = - length_of_MLE // 2
        positionChirpEnd -= length_of_MLE // 2
        BER = []

        for i in range(length_of_MLE):

            positionChirpEnd += 1
            bias += 1
            ofdmBlockStart = positionChirpEnd
            ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
            dataEnd = ofdmBlockEnd + 4 * (N + CP)  # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

            hest, _ = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
            hest_chirp = Hest_from_chirp(y, plot=False, bias=bias)
            hest_chirp = hest_chirp / np.linalg.norm(hest_chirp)

            if plot:
                plot_frequency_response(hest_chirp)
                plot_waveform(ifft(hest_chirp)[:3000])

            # Combine with chirp estimation
            hest = hest / np.linalg.norm(hest)

            if plot:
                plot_frequency_response(hest)
                plot_waveform(ifft(hest)[:3000])

            hest = (1-chirpimportance) * hest + chirpimportance * hest_chirp

            equalizedSymbols, _ = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance=0.5, pilotValues=True)
            outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

            dataToCsv = outputData.ravel()[0:len(ba)]
            demodulatedOutput = ''.join(str(e) for e in dataToCsv)

            ber = calculateBER(ba, dataToCsv)

            for i in range(len(dataToCsv)):
                # Bayesian
                # estimated_bits[i] = estimated_bits[i] + dataToCsv[i] / ber

                # MLE
                estimated_bits[i] = estimated_bits[i] + dataToCsv[i]

            print("Bit Error Rate:" + str(ber))
            BER.append(ber)

        total = 0.0
        for i in range(length_of_MLE):
            total += (1 / BER[i])

        for i in range(len(estimated_bits)):
            # Bayesian
            # estimated_bits[i] = estimated_bits[i] / total

            # MLE
            estimated_bits[i] = estimated_bits[i] / length_of_MLE

            if estimated_bits[i] >= 0.5:
                estimated_bits[i] = 1
            else:
                estimated_bits[i] = 0

        print(estimated_bits)
        # Minimum BER position
        BER = np.array(BER)
        max_position = np.where(BER == np.amin(BER))[0][0] - length_of_MLE // 2 + detected_position
        if plot:
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

        hest, _ = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)

        # Combine with chirp estimation
        hest = hest / np.linalg.norm(hest)
        hest = (1 - chirpimportance) * hest + chirpimportance * hest_chirp

        equalizedSymbols, _ = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance, pilotValues)
        outputData, hardDecision = demapping(equalizedSymbols, demappingTable)

        dataToCsv = outputData.ravel()[0:len(ba)]
        demodulatedOutput = ''.join(str(e) for e in dataToCsv)

        ber = calculateBER(ba, dataToCsv)

        print("Bit Error Rate at minimum ber point:" + str(ber))

        # Maximum Likelihood Estimate
        estimated_bits = np.array(estimated_bits)
        dataToCsv = estimated_bits.ravel()[0:len(ba)]
        demodulatedOutput = ''.join(str(e) for e in dataToCsv)

        print(text_from_bits(demodulatedOutput))
        ber = calculateBER(ba, dataToCsv)

        print("Maximum Likelihood Bit Error Rate:" + str(ber))

    else:
        ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
        dataEnd = ofdmBlockEnd + 4 * (N + CP)  # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

        hest, _ = channel_estimate_known_ofdm(receivedSound[positionChirpEnd: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)

        # Combine with chirp estimation
        hest = hest / np.linalg.norm(hest)
        hest = (1 - chirpimportance) * hest + chirpimportance * hest_chirp

        if plot:
            plt.semilogy(np.arange(N), abs(hest), label='Estimated H via known OFDM')
            plt.grid(True);
            plt.xlabel('Carrier index');
            plt.ylabel('$|H(f)|$');
            plt.legend(fontsize=10)
            plt.show()

        hestImpulse = np.fft.ifft(hest)[0:N // 2]

        if plot:
            plt.plot(np.arange(300), hestImpulse[0:300])
            plt.title('Impulse response')
            plt.xlabel("Sample number")
            plt.ylabel("Impulse response magnitude")
            plt.show()

        equalizedSymbols, _ = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance, pilotValues)
        outputData, hardDecision = demapping(equalizedSymbols, demappingTable)
        z = np.arange(N // 2 - 1)

        if plot:
            plt.scatter(equalizedSymbols[0:N // 2 - 1].real, equalizedSymbols[0:N // 2 - 1].imag, c=z, cmap="bwr")
            plt.show()

        if plot:
            for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
                plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
                plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
                plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
            plt.show()

        dataToCsv = outputData.ravel()[0:len(ba)]
        demodulatedOutput = ''.join(str(e) for e in dataToCsv)
        print(text_from_bits(demodulatedOutput))
        ber = calculateBER(ba, dataToCsv)

        print("Bit Error Rate:" + str(ber))

    return text_from_bits(demodulatedOutput), ber
