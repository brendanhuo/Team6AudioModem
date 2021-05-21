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
import soundfile as sf


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


def H_method_1(h, fs, f1, f2, y):

    H = fft(h, len(y))
    H = H / np.linalg.norm(H)
    H_inv = []

    ratio = round(len(H) / fs)
    print(len(H), f1, f1 / fs * len(H))

    for i in range(round(len(H) / 2)):
        if (f1 / fs) * len(H) <= i <= (f2 / fs) * len(H):
            H_inv.append(1 / H[i])
        else:
            H_inv.append(0.0)

    H_back = H_inv[::-1]
    for item in H_back:
        H_inv.append(item)

    H_inv = np.array(H_inv)

    return smooth(H_inv, 10)


def H_method_2(position, y, x, samples, f1, f2, fs):

    noisy_signal = y[(position - len(x)):(position + samples)]
    Y = fft(noisy_signal, len(y))
    x_f = []

    for i in range(len(x) + samples):
        if i >= len(x):
            x_f.append(0.0)
        else:
            x_f.append(x[i])

    x = np.array(x_f)
    X = fft(x, len(y))
    H_0 = X / Y
    H = []
    for i in range(round(len(H_0) / 2)):
        if (f1 / fs) * len(H_0) <= i <= (f2 / fs) * len(H_0):
            H.append(H_0[i])
        else:
            H.append(0.0)

    H_back = H[::-1]

    for item in H_back:
        H.append(item)

    H = np.array(H)

    return H


def flatten(signal):
    y = []

    for i in range(len(signal)):
        y.append(signal[i][0])

    # Format
    y = np.array(y)

    return y


def chirp_estimator_(listening_time, samples=2000, f1=60.0, f2=6000.0, T=5.0, fs=44100):
    """Listens and determines start point, Impulse response, and sample at which t occurs relative to recording start time"""

    # Noise reduction
    # print("Measuring noise")

    # Define r as output array
    # noise = sd.rec(int(5 * fs), samplerate=fs, channels=1)
    # sd.wait()  # Wait until recording is finished

    # noise is multidimensional, so create 1D array, y
    # n = []

    #for i in range(len(noise)):
     #   n.append(noise[i][0])

    # Format
    #n = np.array(n)
    #n = n / np.linalg.norm(n)

    #N = fft(n, listening_time * fs)
    #plt.semilogy(np.linspace(0, fs * 0.5, round(len(N) / 2), False), smooth(abs(N), 100)[0:round(len(N) / 2)])
    #plt.title('Noise Frequency Spectrum')
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Relative Amplitude')
    #plt.grid(True)
    #plt.show()

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

    # Format
    y = np.array(y)

    print('Finish Recording')

    # Normalise
    y = y / np.linalg.norm(y)

    # Remove Noise
    #Y = fft(y)
    #Y = Y * N
    #y = ifft(Y)
    #print(y)
    #y = np.real(y)

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
    h = h / np.linalg.norm(h)

    # View results

    # play(y)
    # play(h)

    out_h = 'h.wav'
    wavf.write(out_h, fs, h)
    plt.plot(np.linspace(0, len(h) / fs, len(h), False), smooth(h, 1))
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

    H = H_method_1(h, fs, f1, f2, y)
    # H = H_method_2(position, y, x, samples, f1, f2, fs)
    # H = H_test(fs, f1, f2, y)

    # H_inv = smooth(H_inv, 20)
    # plt.semilogy(np.linspace(0, fs * 0.5, round(len(H)/2), False), smooth(abs(H), 100)[0:round(len(H)/2)])
    # plt.semilogy(np.linspace(0, fs * 0.5, round(len(H_inv) / 2), False), smooth(abs(H_inv), 100)[0:round(len(H_inv) / 2)])
    # h = np.real(ifft(H)[0:round(samples / 2)])

    #plt.title('Frequency Fresponse')
    #plt.xlabel('Frequency (Hz)')
    #plt.ylabel('Relative Amplitude')
    #plt.grid(True)
    #plt.show()

    #plt.plot(np.linspace(0, len(h) / fs, len(h), False), h)
    #plt.title('Impulse Response')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Relative Amplitude')
    #plt.grid(True)
    #plt.show()

    Y = fft(y)
    X = Y / H
    x = ifft(X)

    x = np.real(x)
    x = x / np.linalg.norm(x)
    # x, remainder = signal.deconvolve(y[(position - round(T*fs)):position], h[1:])

    plot_spectrogram(x, 44100)

    plt.plot(np.linspace(0, len(x) / fs, len(x), False), x)

    plt.title('Recovered Smoothed Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()
    play(y[(position - round(T*fs)):])
    play(x[(position - round(T*fs)):])

    return h, position


def test(smoothing_factor):

    y, _ = sf.read('voice.wav', dtype='float32')
    h, _ = sf.read('h_average.wav', dtype='float32')

    y = y / np.linalg.norm(y)
    h = h / np.linalg.norm(h)

    Y = fft(y)
    H = smooth(fft(h, len(Y)), smoothing_factor * 100)
    # H = smooth(fft(h, len(Y)), 1000)
    X = Y / H
    x = ifft(X)
    x = np.real(x)
    x = x / np.linalg.norm(x)

    plt.plot(np.linspace(0, len(x) / 44100, len(x), False), x)

    plt.title('Recovered Smoothed Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()

    play(x)

    plot_frequency_response(H)
    # plot_spectrogram(x, 44100)

    array_to_wav(x, "smoothed_output.wav")


def plot_waveform(x, name, fs=44100):

    plt.plot(np.linspace(0, len(x) / fs, len(x), False), x)

    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()


def plot_frequency_response(H, fs=44100):

    fig, axs = plt.subplots(2, 1)
    axs[0].semilogy(np.linspace(0, fs * 0.5, round(len(H) / 2), False), smooth(abs(H), 1)[0:round(len(H) / 2)])
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Relative Amplitude')

    axs[1].plot(np.linspace(0, fs * 0.5, round(len(H) / 2), False), smooth(np.angle(H, deg=True), 100)[0:round(len(H) / 2)])
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (degrees)')

    plt.grid(True)
    fig.tight_layout()
    plt.show()


def chirp_2(listening_time=12, samples=10000, f1=50.0, f2=7000.0, T=1, fs=44100, time_between=1, number_chirps=3, simulate=False, filename=""):

    # Create chirp
    x = exponential_chirp(T)

    x_chain = []

    for i in range(round((len(x) + fs * time_between) * number_chirps)):
        if i % round((len(x) + time_between * fs)) < len(x):
            x_chain.append(x[i % round((len(x) + time_between * fs))])
        else:
            x_chain.append(0.0)

    x_chain = np.array(x_chain)
    plot_waveform(x_chain, "Multichirp waveform", 44100)

    # For convenience, save as wav file: occasionally, python deletes x for no apparent reason - redefine x in case
    out_x_chain = 'chirp_chain.wav'
    wavf.write(out_x_chain, fs, x_chain)

    # Create normalised time-reversed signal
    x_r = x[::-1]

    # Apply exponential envelope and normalise
    for i in range(len(x_r)):
        x_r[i] = x_r[i] * math.e ** (-(i / (T * fs)) * math.log2(f2 / f1))

    x_r = x_r / np.linalg.norm(x_r)

    if not simulate:
        print('Recording')

        # Define r as output array
        r = sd.rec(int(listening_time * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, r)  # Save as WAV file

        # r is multidimensional, so create 1D array, y
        y = []

        for i in range(len(r)):
            y.append(r[i][0])

        # Format
        y = np.array(y)
        print('Finish Recording')

    else:
        y, _ = sf.read(filename, dtype='float32')

    # Normalise
    y = y / np.linalg.norm(y)

    # Convolve output with x_r and normalise
    h = signal.fftconvolve(x_r, y)
    h = h / np.linalg.norm(h)

    plot_waveform(h, "Convoluted Signal")

    h_copy = h

    positions = []

    for i in range(number_chirps):
        if i == 0:
            positions.append(np.where(h_copy == np.amax(h_copy))[0][0])
            h_copy = np.delete(h_copy, positions[0])
        else:
            while True:
                test = np.where(h_copy == np.amax(h_copy))[0][0]
                h_copy = np.delete(h_copy, test)

                j = 0
                for item in positions:
                    if abs(test - item) > time_between * fs * 0.5:
                        j += 1

                if j == len(positions):
                    positions.append(test)
                    break

    positions.sort()
    print(positions)

    estimated_positions = []
    print(positions[2] - positions[1], positions[1] - positions[0])

    total = 0
    for item in positions:
        total += item

    centre = round(total / len(positions))
    estimated_positions.append(centre)

    if len(positions) % 2 == 0:
        for i in range(round(len(positions) / 2)):
            estimated_positions.append(round(centre + (i + 0.5) * (T + time_between) * fs))
            estimated_positions.append(round(centre - (i + 0.5) * (T + time_between) * fs))
    else:
        for i in range(round((len(positions) - 1) / 2)):
            estimated_positions.append(round(centre + (i + 1) * (T + time_between) * fs))
            estimated_positions.append(round(centre - (i + 1) * (T + time_between) * fs))

    estimated_positions.sort()
    print(estimated_positions)

    h_list = []

    for i in range(len(positions)):
        h_list.append(h[positions[i]:(positions[i] + samples)])
        plot_waveform(h_list[i], "waveform {}".format(i + 1))

    H_list = []

    for i in range(len(h_list)):
        H_list.append(fft(h_list[i]))

    H_average = []

    for i in range(len(H_list[0])):
        total = 0.0
        for H in H_list:
            total += H[i]
        H_average.append(total / len(H_list))

    plot_frequency_response(smooth(H_average, 10))

    # H_average = smooth(H_average, 5)

    h_average = np.real(ifft(H_average))
    array_to_wav(h_average, "h_average.wav")

    plot_waveform(h_average, "h averaged")


# Demonstration
# array_to_wav(exponential_chirp(T=5, f1=40, f2=10000.0, window_strength=10.0, fs=44100), "5_seconds_large_range.wav")
# chirp_estimator(samples=20000, f1=60.0, f2=6000.0, T=3, fs=44100, listening_time=7)
# chirp_estimator(samples=20000, f1=60.0, f2=6000.0, T=3, fs=44100, listening_time=5)
# chirp_estimator(samples=20000, f1=50.0, f2=7000.0, T=10, fs=44100, listening_time=15)
# chirp_2(simulate=True, filename="TASCAM_0112_noise_reduced.wav")
test(smoothing_factor=200)

