import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def plot_spectrogram(array, fs=44100):
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


def plot_waveform(x, fs=44100):
    """Plots waveform from an array"""

    plt.plot(np.linspace(0.0, float(len(x) / fs), len(x), False), x)

    plt.title("waveform")
    plt.xlabel('Time (s)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.show()


def plot_frequency_response(H, fs=44100):
    """Plots frequency response (magnitude and phase) from an array"""

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
