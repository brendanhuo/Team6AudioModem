import simpleaudio as sa
import pyaudio
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write


def play(array, fs=44100):
    audio = array * (2 ** 15 - 1) / np.max(np.abs(array))
    audio = audio.astype(np.float32)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

def playrec(array, fs = 44100):
    audio = array * (2 ** 15 - 1) / np.max(np.abs(array))
    audio = audio.astype(np.float32)
    myrecording = sd.playrec(audio, fs, channels=1)
    print("recording")
    print(sd.default.device)
    sd.wait()  # Wait until recording is finished
    print("writing")
    write('test_recording.wav', fs, myrecording)  # Save as WAV file 
    print("done")
    return myrecording

def sound(array, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=len(array.shape), rate=fs, output=True)
    stream.write(array.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def record(seconds):
    myrecording = sd.rec(int(seconds * sd.default.samplerate))
    print("recording")
    print(sd.default.device)
    sd.wait()  # Wait until recording is finished
    print("done")
    return myrecording

def audioDataFromFile(filename):
    data, fs = sf.read(filename, dtype='float32')  
    return data






