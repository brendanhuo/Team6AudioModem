from globals import *
from binary_utils import *
from transmitter import *
from receiver import *
from chirp import *
from channel import *
from graphing_utils import *
from audio_utils import *
from scipy import fft, ifft


def Hest_from_chirp(y, plot=False):
    """Estimates impulse response from chirp, and returns frequency response"""

    # Create chirp
    x = exponential_chirp()

    # Create normalised time-reversed signal
    x_r = x[::-1]

    # Apply exponential envelope and normalise
    for i in range(len(x_r)):
        x_r[i] = x_r[i] * math.e ** (-(i / (chirp_length * fs)) * math.log2(f2 / f1))

    x_r = x_r / np.linalg.norm(x_r)

    # Normalise
    y = y / np.linalg.norm(y)

    # Convolve output with x_r and normalise
    h = signal.fftconvolve(x_r, y)
    h = h / np.linalg.norm(h)

    # plot_waveform(h, "Convoluted Signal")

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
        if plot:
            plot_waveform(h_list[i], "waveform {}".format(i + 1))

    H_list = []

    for i in range(len(h_list)):
        H_list.append(fft(h_list[i], N))

    H_average = []

    for i in range(len(H_list[0])):
        total = 0.0
        for H in H_list:
            total += H[i]
        H_average.append(total / len(H_list))

    plot_frequency_response(smooth(H_average, 10))

    H_average = smooth(H_average, smoothing_factor)

    h_average = np.real(ifft(H_average))

    if plot:
        plot_waveform(h_average, "h averaged")

    return H_average
