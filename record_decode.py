from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 

# # Comment out if not recording
# print('Recording')

# listening_time = 7
# r = sd.rec(int(listening_time * fs), samplerate=fs, channels=1)
# sd.wait()  # Wait until recording is finished
# write('audio/sound4.wav', fs, r)  # Save as WAV file

# audio_received = []
# for i in range(len(r)):
#     audio_received.append(r[i][0])

# np.save("audio/sound4.npy", np.asarray(audio_received))

with open("./text/lorem.txt") as f:
    contents = f.read()

ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())
print(len(ba))

dataCarriers, pilotCarriers = assign_data_pilot(K, P)

receivedSound = np.load("audio/sound1.npy")
plt.plot(receivedSound)
plt.show()

positionChirpStart = chirp_synchroniser(receivedSound, chirpLength)

# OFDM block channel estimation
ofdmBlockStart = positionChirpStart + chirpLength * fs
ofdmBlockEnd = positionChirpStart + chirpLength * fs + (N + CP) * blockNum
dataEnd = ofdmBlockEnd + 4 * (N + CP) # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

hest = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
plt.plot(np.arange(N), abs(hest), label='Estimated H via cubic interpolation')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

hestImpulse = np.fft.ifft(hest)
plt.plot(np.arange(N), hestImpulse)
plt.show()

equalizedSymbols = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers)
outputData, hardDecision = demapping(equalizedSymbols , demappingTable)
z = np.arange(N//2-1)
plt.scatter(equalizedSymbols[0:N//2-1].real, equalizedSymbols[0:N//2-1].imag, c=z, cmap="bwr")
plt.show()
for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
plt.show()

dataToCsv = outputData.ravel()
demodulatedOutput = ''.join(str(e) for e in dataToCsv)
print(text_from_bits(demodulatedOutput))
ber = calculateBER(ba, dataToCsv)

print("Bit Error Rate:" + str(ber))
