from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 

# Comment out if not recording
print('Recording')

listening_time = 150
r = sd.rec(int(listening_time * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('audio/image_received_autumn_6_ldpc.wav', fs, r)  # Save as WAV file

audio_received = []
for i in range(len(r)):
    audio_received.append(r[i][0])

np.save("audio/image_received_autumn_6_ldpc.npy", np.asarray(audio_received))
