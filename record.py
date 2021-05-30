from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import * 

# Comment out if not recording
print('Recording')

listening_time = 36
r = sd.rec(int(listening_time * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('audio/outside/asyoulikeit-rec15.wav', fs, r)  # Save as WAV file

audio_received = []
for i in range(len(r)):
    audio_received.append(r[i][0])
#
np.save("audio/outside/asyoulikeit-rec15.npy", np.asarray(audio_received))
