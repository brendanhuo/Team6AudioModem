import simpleaudio as sa
import pyaudio
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np

fs = 44100

sd.default.samplerate = fs  # Sample rate
sd.default.channels = 1


def save(array, filename):
    """Save array as a wav file"""

    write(filename, fs, array)  # Save as WAV file


def record_and_save(filename, seconds):
    """Recording sound and saving it as a wav file"""

    myrecording = sd.rec(int(seconds * sd.default.samplerate))
    print("recording")
    print(sd.default.device)
    sd.wait()  # Wait until recording is finished
    print("writing")
    write(filename, fs, myrecording)  # Save as WAV file 
    print("done")

    return myrecording


def record(seconds):
    """Recording sound"""

    myrecording = sd.rec(int(seconds * sd.default.samplerate))
    print("recording")
    print(sd.default.device)
    sd.wait()  # Wait until recording is finished
    print("done")
    return myrecording


def play_note(note):
    """Play note on computer which is possible using a Bluetooth speaker"""

    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int16)

    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    print("Playing note")
        
    # Wait for playback to finish before exiting
    play_obj.wait_done()
    print("done")


def play(data, fs=fs):
    """Play the recorded data"""

    sd.play(data, fs)
    print("playing")
    print(sd.default.device)
    sd.wait()  # Wait until file is done playing


def playFile(filename):
    """Play the recorded data, except now from an external file"""

    # Extract data and sampling rate from file
    data, fs = sf.read(filename, dtype='float32')  
    sd.play(data, fs)
    print("playing")
    print(sd.default.device)
    sd.wait()  # Wait until file is done playing
    print("done")


def audioDataFromFile(filename):
    """Return the array of values representing the wav file"""

    data, fs = sf.read(filename, dtype='float32')

    return data
