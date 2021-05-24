from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 

# # Comment out if not recording
#print('Recording')

#listening_time = 7
#r = sd.rec(int(listening_time * fs), samplerate=fs, channels=1)
#sd.wait()  # Wait until recording is finished
#write('audio/img_sound1.wav', fs, r)  # Save as WAV file

#audio_received = []
#for i in range(len(r)):
#    audio_received.append(r[i][0])

#np.save("audio/img_sound1.npy", np.asarray(audio_received))

image_path = "./image/bali.tif"
ba = image2bits(image_path, True)
print(len(ba))

dataCarriers, pilotCarriers = assign_data_pilot(K, P)

receivedSound = np.load("audio/img_sound1.npy")
plt.plot(receivedSound)
plt.show()

positionChirpEnd = chirp_synchroniser(receivedSound)

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd
ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
dataEnd = ofdmBlockEnd + 1584 * (N + CP) # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

hest, noise = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
plt.semilogy(np.arange(N), abs(hest), label='Estimated H via known OFDM')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

plt.semilogy(np.arange(N), abs(noise), label='Estimated noise via known OFDM')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|N(f)|$'); plt.legend(fontsize=10)
plt.show()

hestImpulse = np.fft.ifft(hest)[0:N//2]
plt.plot(np.arange(N//2), hestImpulse[0:N//2])
plt.title('Impulse response')
plt.show()

noiseEstimateReal = np.mean(noise.real ** 2)
noiseEstimateImag = np.mean(noise.imag ** 2)

print("Estimated real part of noise: " + str(noiseEstimateReal))
print("Estimated imaginary part of noise: " + str(noiseEstimateImag))

equalizedSymbols, hestAll = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance, pilotValues)

llrsReceived = return_llrs(equalizedSymbols, hestAll, (noiseEstimateReal, noiseEstimateImag))
print(llrsReceived[:100])

outputData, hardDecision = demapping(equalizedSymbols , demappingTable)
z = np.arange(N//2-1)
plt.scatter(equalizedSymbols[0:N//2-1].real, equalizedSymbols[0:N//2-1].imag, c=z, cmap="bwr")
plt.show()
for qpsk, hard in zip(equalizedSymbols[0:400], hardDecision[0:400]):
    plt.plot([qpsk.real, hard.real], [qpsk.imag, hard.imag], 'b-o');
    plt.plot(hardDecision[0:400].real, hardDecision[0:400].imag, 'ro')
plt.grid(True); plt.xlabel('Real part'); plt.ylabel('Imaginary part'); plt.title('Demodulated Constellation');
plt.show()

dataToCsv = outputData.ravel()[:len(ba)]

ber = calculateBER(ba, dataToCsv)
print("Bit Error Rate:" + str(ber))

byte_array = []
for i in range (len(dataToCsv)//8):
    demodulatedOutput = ''.join(str(e) for e in dataToCsv[8*i:8*(i+1)])
    byte_array.append(int(demodulatedOutput,2))
plt.imshow(np.array(byte_array)[0:1418100].reshape(489, 725, 4))
plt.title('Image received')
plt.show()


        


