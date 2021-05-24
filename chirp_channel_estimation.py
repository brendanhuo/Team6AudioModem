from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from graphing_utils import *
from audio_utils import *
from scipy import fft, ifft
import scipy.signal as sig


def Hest_from_chirp(y, plot=False, bias=0):
    """Estimates impulse response from chirp, and returns frequency response"""

    T = chirp_length

    # Create chirp
    x = exponential_chirp()

    if plot:
        plot_waveform(x)
        plot_frequency_response(fft(x))

    # Create normalised time-reversed signal
    x_r = x[::-1]

    # Apply exponential envelope and normalise
    for i in range(len(x_r)):
        x_r[i] = x_r[i] * math.e ** (-(i / (chirp_length * fs)) * math.log2(f2 / f1))

    x_r = x_r / np.linalg.norm(x_r)

    if plot:
        plot_waveform(x_r)
        plot_frequency_response(fft(x_r))

    # Normalise
    y = y / np.linalg.norm(y)

    ir = sig.fftconvolve(x, x_r, mode='same')

    # Convolve output with x_r and normalise
    h = signal.fftconvolve(x_r, y)
    h = h / np.linalg.norm(h)

    if plot:
        plot_waveform(h)
        plot_waveform(ir)

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

    estimated_positions = []

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

    h_list = []

    for i in range(len(estimated_positions)):
        h_temp = h[estimated_positions[i] - bias:(estimated_positions[i] - bias + N // 2)]
        h_temp = np.array(h_temp)
        h_back = h_temp[::-1]
        h_temp = np.append(h_temp, h_back)
        h_temp = h_temp.tolist()
        h_list.append(h_temp)

        if plot:
            plot_waveform(h_list[i])

    H_list = []

    for i in range(len(h_list)):
        H_list.append(fft(h_list[i], N))

    H_average = []

    for i in range(len(H_list[0])):
        total = 0.0
        for H in H_list:
            total += H[i]
        H_average.append(total / len(H_list))

    if plot:
        plot_frequency_response(smooth(H_average, 1))

    H_average = smooth(H_average, smoothing_factor)

    h_average = np.real(ifft(H_average))

    if plot:
        plot_waveform(h_average)

    if method_2:
        H_average = [0.0]*N

        for i in range(len(estimated_positions)):
            H_temp = H_method_2(estimated_positions[i], y, exponential_chirp())

            for j in range(N):
                H_average[j] += H_temp[j]

    H_average = np.array(H_average)

    return H_average


def H_method_2(position, y, x):

    noisy_signal = y[(position - len(x)):position]
    noisy_signal = noisy_signal / np.max(noisy_signal)
    x = x / np.max(x)

    h = ifft(fft(noisy_signal) / fft(x))
    H = fft(h, N)

    return smooth(H, smoothing_factor)
