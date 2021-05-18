import wave
import math
import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa
from scipy import fft, ifft
import scipy.io.wavfile as wavf
from scipy import signal
import sounddevice as sd
from scipy.io.wavfile import write


def open_wav(file):
    """Opens file and converts to array"""

    # Read file to get buffer
    ifile = wave.open(file)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)

    # Convert buffer to float32 using NumPy
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0
    max_int16 = 2 ** 15
    audio_normalised = audio_as_np_float32 / max_int16

    return audio_normalised


def plot_spectrogram(array, fs):
    """Plots spectrogram from array"""

    f, t, Sxx = signal.spectrogram(array, fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def smooth(y, box_pts):
    """Returns smoothed response"""

    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')

    return y_smooth


def array_to_wav(array, filename):
    """Converts array to Wav file"""
    fs = 44100
    wavf.write(filename, fs, array)


def play(array, fs=44100):
    """Plays audio file"""

    # Normalise
    audio = array * (2 ** 15 - 1) / np.max(np.abs(array))
    audio = audio.astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()


def exponential_chirp(T=10.0, f1=60.0, f2=6000.0, window_strength=10.0, fs=44100):
    """Produces chirp and returns impulse characteristics"""

    t_list = np.linspace(0, T, int(round(T * fs)), False)
    profile = []
    r = f2/f1

    # Calculate Sine Sweep time domain values
    for t in t_list:
        value = math.sin(2*math.pi*T*f1*((r**(t/T)-1)/(math.log(r, math.e))))*(1-math.e**(-window_strength*t))*(1-math.e**(window_strength*(t-T)))
        profile.append(value)

    # Format
    profile = np.array(profile)

    return profile


def chirp_estimator(listening_time, samples=2000, f1=60.0, f2=6000.0, T=5, fs=44100):
    """Listens and determines start point, Impulse response, and sample at which t occurs relative to recording start time"""

    # Create chirp, and time reversed signal x_r
    x = exponential_chirp(T)
    x_r = x[::-1]

    # Apply exponential envelope and normalise
    for i in range(len(x_r)):
        x_r[i] = x_r[i] * math.e ** (-(i / (T * fs)) * math.log2(f2 / f1))

    x_r = x_r / np.linalg.norm(x_r)

    # For convenience, save as wav file: occasionally, python deletes x for no apparent reason - redefine x in case
    out_x = 'chirp.wav'
    x = exponential_chirp(T)
    wavf.write(out_x, fs, x)

    print('Recording')

    # Define r as output array
    r = sd.rec(int(listening_time * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, r)  # Save as WAV file

    # r is multidimensional, so create 1D array, y
    y = []

    for i in range(len(r)):
        y.append(r[i][0])

    print('Finish Recording')

    # Format and normalise
    y = np.array(y)
    y = y / np.linalg.norm(y)

    # Convolve output with x_r
    h = signal.fftconvolve(x_r, y)

    # Estimate Impulse Response start point
    position = np.where(h == np.amax(h))[0][0]

    # Find closest 0 to estimate above and set as Impulse Response start point
    i = 1
    while True:
        if 0 <= h[position - i] <= h[position - i + 1]:
            i += 1
        else:
            break
    position -= i
    if abs(h[position]) > abs(h[position - 1]):
        position -= 1
    h[position] = 0

    # Duplicate and truncate h
    h_0 = h
    h = h[position:int(position + samples)]

    # View results

    play(y)
    play(h)

    out_h = 'h.wav'
    wavf.write(out_h, fs, h)
    plt.plot(np.linspace(0, len(h) / fs, len(h), False), h)
    plt.title('Measured Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()

    plt.plot(np.linspace(0, len(h_0) / fs, len(h_0), False), h_0)
    plt.title('Full Recording')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()

    # Find Frequency Response
    H = fft(h)
    plt.semilogy(np.linspace(0, fs, len(H), False), abs(H))
    plt.grid(True)
    plt.show()

    return h, H, position


# Demonstration
chirp_estimator(samples=10000, f1=60.0, f2=6000.0, T=3, fs=44100, listening_time=7)
