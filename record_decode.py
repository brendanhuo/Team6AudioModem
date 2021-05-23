from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from graphing_utils import *
from audio_utils import *
from global_functions import *

# Original Text
with open("./text/lorem.txt") as f:
    contents = f.read()

# Received Audio
ba = bitarray.bitarray()
ba.frombytes(contents.encode('utf-8'))
ba = np.array(ba.tolist())
receivedSound = audioDataFromFile("audio/sound3.wav")

# wav_transmission(ba, "audio/output.wav", plot=True)

print(decode_and_compare_text(receivedSound, ba, False))
