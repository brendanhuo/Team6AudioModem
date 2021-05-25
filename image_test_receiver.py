from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 

image_path = "./image/autumn.tif"
ba, image_shape = image2bits(image_path, True)
print(len(ba))

dataCarriers, pilotCarriers = assign_data_pilot(K, P, bandLimited = True)

ba = np.append(ba, np.zeros(len(dataCarriers)*2 - (len(ba) - len(ba)//mu//len(dataCarriers) * len(dataCarriers) * mu)))
bitsSP = ba.reshape((len(ba)//mu//len(dataCarriers), len(dataCarriers), mu))
numOFDMblocks = len(bitsSP)

receivedSound = np.load("audio/image_received_autumn_2.npy")
plt.plot(receivedSound)
plt.show()

positionChirpEnd = chirp_synchroniser(receivedSound) -23

# OFDM block channel estimation
ofdmBlockStart = positionChirpEnd
ofdmBlockEnd = positionChirpEnd + (N + CP) * blockNum
dataEnd = ofdmBlockEnd + numOFDMblocks * (N + CP) # 4 is the number of data OFDM blocks we are sending, should be determined by metadata

hest = channel_estimate_known_ofdm(receivedSound[ofdmBlockStart: ofdmBlockEnd], seedStart, mappingTable, N, K, CP, mu)
plt.semilogy(np.arange(N), abs(hest), label='Estimated H via known OFDM')
plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
plt.show()

hestImpulse = np.fft.ifft(hest)[0:N//2]
plt.plot(np.arange(N//2), hestImpulse[0:N//2])
plt.title('Impulse response')
plt.show()

<<<<<<< HEAD
equalizedSymbols, _ = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance = 0.5, pilotValues = True)
=======
equalizedSymbols = map_to_decode(receivedSound[ofdmBlockEnd:dataEnd], hest, N, K, CP, dataCarriers, pilotCarriers, pilotValue, pilotImportance = 0.3, pilotValues = True)
>>>>>>> 0df71e032c4e34f91ecdceea8630d28d7be09e4e
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
plt.show()
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


        


