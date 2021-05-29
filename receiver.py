import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import simpleaudio as sa
import pyaudio
import sounddevice as sd
import soundfile as sf
import math, cmath
from scipy.io.wavfile import write
from transmitter import *
from chirp import *
from numpy.random import default_rng
from scipy import signal
from sklearn.linear_model import LinearRegression
from math import floor
from scipy.signal import savgol_filter

def removeCP(signal, CP, N):
    """Removes cyclic prefix"""

    return signal[CP:(CP+N)]


def DFT(ofdmData, N):
    """Takes DFT"""

    return np.fft.fft(ofdmData, N)


def remove_zeros(position, data):
    """Removes preappended zeros from data"""

    return data[position:]

def get_channel_offset(unwrapped_angle):
    
    y = unwrapped_angle
    x = np.arange(len(y))
    
    model = LinearRegression().fit(x[:, np.newaxis], y)
    y_fit = model.predict(x[:, np.newaxis])
    
    slope = model.coef_[0]
    #yhat = savgol_filter(y, 31, 1) # window size 11, polynomial order 1
    #plt.plot(y)
    #plt.plot(y_fit)
    #plt.plot(yhat)
    #plt.plot(np.diff(yhat))
    #plt.show()
    return -slope * N / 2 / np.pi, y_fit

def get_continue_seq(str_list):
    ls = str_list
    len_ls = len(ls)
    index_count = {}
    for x in range(len_ls):
        key = index_count.keys()
        if key:
            diff_x1 = ls[x]-ls[x-1]
            if diff_x1==1:
                index_count[max(key)].append(ls[x])
            else:
                index_count[x] = [ls[x]]
        else:
            index_count[x] = [ls[x]]
    res = index_count[max(index_count, key=lambda x: len(index_count[x]))]
    return res

def channel_estimate_known_ofdm(knownOFDMBlock, randomSeedStart, mappingTable, N, K, CP, mu):
    """Channel estimate using known OFDM block symbols"""

    numberOfBlocks = len(knownOFDMBlock) // (N+CP)
    hestAtSymbols = np.zeros(N, dtype = complex) # estimate of the channel gain at particular frequency bins
    offsets = []
    for i in range(numberOfBlocks):
        # Retrace back the original seed
        rng = default_rng(randomSeedStart + i)
        bits = rng.binomial(n=1, p=0.5, size=((K-1)*2))
        bitsSP = bits.reshape(len(bits)//mu, mu)
        symbol = np.array([mappingTable[tuple(b)] for b in bitsSP])
        expectedSymbols = np.append(np.append(0, symbol), np.append(0,np.conj(symbol)[::-1]))

        receivedSymbols = knownOFDMBlock[i*(N+CP): (i+1)*(N+CP)]
        receivedSymbols = removeCP(receivedSymbols, CP, N)
        receivedSymbols = DFT(receivedSymbols, N)

        # hestAtSymbols = (hestAtSymbols * i + (receivedSymbols / expectedSymbols)) / (i + 1) # Averaging over past OFDM blocks
        for j in range(N):
            if j == N//2 or j == 0:
                hestAtSymbols[j] = 0
            else:
                div = (receivedSymbols[j]/expectedSymbols[j])
                hestAtSymbols[j] = (hestAtSymbols[j] * i + div) / (i + 1) # Average over past OFDM blocks
                #hestAtSymbols[j] = div
        #offsets.append(get_channel_offset(np.unwrap(np.angle(hestAtSymbols)))[0])
    yhat = savgol_filter(np.unwrap(np.angle(hestAtSymbols)), 31, 1) # window size 11, polynomial order 1
    #second_deriv = np.gradient(np.gradient(np.unwrap(np.angle(hestAtSymbols))[0:N//2]))
    second_deriv = np.gradient(np.gradient(yhat[0:N//2]))
    boundary = np.mean(abs(second_deriv))
    goodFrequencies = []
    for i in range(len(second_deriv)):
        if abs(second_deriv[i]) <= boundary:
            goodFrequencies.append(i)
    best_sequence = get_continue_seq(goodFrequencies)
    #unwrapped_angle = np.unwrap(np.angle(hestAtSymbols))
    unwrapped_angle = yhat
    offset, y_fit = get_channel_offset(unwrapped_angle[best_sequence])
    plt.plot(unwrapped_angle[best_sequence], label = 'unwrapped angle at straight section')
    plt.plot(y_fit, label = 'linear regression')
    plt.legend()
    plt.show()

    plt.plot(np.unwrap(np.angle(hestAtSymbols)), label = 'Hest')
    #plt.plot(np.gradient(np.unwrap(np.angle(hestAtSymbols))), label = 'first deriv')
    #plt.plot(second_deriv, label = 'second deriv')
    #plt.hlines(boundary, xmin = 0, xmax = 8000, label = 'boundary')
    plt.legend()
    plt.show()
    if len(best_sequence) <= 30:
        return hestAtSymbols, 0
    return hestAtSymbols, offset

def channel_estimate_pilot(ofdmReceived, pilotCarriers, pilotValue, N):
    """Performs channel estimation using pilot values"""

    pilotsPos = ofdmReceived[pilotCarriers]  # extract the pilot values from the RX signal
    pilotsNeg = ofdmReceived[-pilotCarriers]
    hestAtPilots1 = pilotsPos / pilotValue # divide by the transmitted pilot values
    hestAtPilots2 = pilotsNeg / np.conj(pilotValue)
    
    hestAtPilots = np.append(hestAtPilots1, hestAtPilots2).ravel()
    # Perform interpolation between the pilot carriers to get an estimate
    # of the channel in the data carriers. Here, we interpolate absolute value and phase 
    # separately
    hestAbs = scipy.interpolate.interp1d(np.append(pilotCarriers, N - pilotCarriers).ravel(), abs(hestAtPilots), kind='cubic', fill_value="extrapolate")(np.arange(N))
    hestPhase = scipy.interpolate.interp1d(np.append(pilotCarriers, N - pilotCarriers).ravel(), np.angle(hestAtPilots), kind='cubic', fill_value="extrapolate")(np.arange(N))
    hest = hestAbs * np.exp(1j*hestPhase)

    #plt.figure(1)
    #plt.plot(hestPhase)
    #plt.title('No unwrap')
    #plt.figure(2)
    #plt.plot(np.unwrap(hestPhase))
    #plt.title('With unwrap')
    #plt.show()
    return hest


def map_to_decode(audio, channelH, N, K, CP, dataCarriers, pilotCarriers, pilotValue, offset, offsets, samplingMismatch, pilotImportance = 0, pilotValues = True):
    """Builds demodulated constellation symbol from OFDM symbols"""

    dataArrayEqualized = []
    Hest = channelH
    np.save("testing_angles.npy", np.asarray(np.angle(Hest)))
    HestAggregate = []

    offset_old = floor(np.mean(offsets))
    remainingDifference = np.mean(offsets) - offset_old

    for i in range(len(audio)//(N+CP)):
        remainingDifference += i * samplingMismatch
        if abs(remainingDifference)>=1:
            data = audio[i*(N+CP)+floor(remainingDifference): (N+CP)*(i+1)+floor(remainingDifference)]
            remainingDifference = remainingDifference - floor(remainingDifference)
        else:
            data = audio[i*(N+CP): (N+CP)*(i+1)]
        data = removeCP(data, CP, N)
        data = DFT(data, N)
        for l in range(len(data)):
            #data[l] *= cmath.exp(remainingDifference * l * 2 * np.pi/N)
            data[l] *= cmath.exp(offset * l * 2 * np.pi/N)
        if pilotValues:
            pilotHest = channel_estimate_pilot(data, pilotCarriers, pilotValue, N)
        else:
            pilotHest = 0
        # Hest = channelEstimate(data)
        Hest = (1-pilotImportance) * Hest + pilotImportance*pilotHest
        data_equalized = data/Hest
        dataArrayEqualized.append(data_equalized[0:K][dataCarriers])

        HestAggregate.append(Hest[0:K][dataCarriers])

    return np.array(dataArrayEqualized).ravel(), np.array(HestAggregate).ravel()


def find_location_with_pilot(approxLocation, data, rangeOfLocation, pilotValue, pilotCarriers, N, CP):
    """Performs fine synchronization using pilot symbols (NOT USED)"""

    minValue = 100000

    for i in range(-rangeOfLocation+1, rangeOfLocation):

        ofdmSymbol = data[approxLocation + i: approxLocation+i+N+CP]
        ofdmNoCP = removeCP(ofdmSymbol, CP. N)
        ofdmTime = DFT(ofdmNoCP)
        angleValues = np.angle(ofdmTime[pilotCarriers]/pilotValue)
        # plt.plot(pilotCarriers, angle_values, label = i)
        
        absSlope = abs(np.polyfit(pilotCarriers,angleValues,1)[0])
        if absSlope < minValue:
            minValue = absSlope
            bestLocation = i + approxLocation
            #print(minValue)

    return bestLocation, minValue


def chirp_synchroniser(audio):
    """Performs synchronization using a chirp, returning the position of the END of the chirp chain signal"""

    T = chirp_length

    # chirp_function is the chirp function specified to match filter with
    x = exponential_chirp()
    x_r = x[::-1]

    # Format and normalise
    y = audio

    # Convolve output with x_r and normalise
    h = signal.fftconvolve(x_r, y)
    h = h / np.linalg.norm(h)
    plot_waveform(h)

    if number_chirps == 1:

        # Estimate start point if one chirp
        position = np.where(h == np.amax(h))[0][0] + time_between * fs

    else:

        # plot_waveform(h, "Convoluted Signal")
        h_copy = h
        positions = []

        # determine chirp chain positions
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

        # determine average position
        total = 0
        for item in positions:
            total += item

        centre = round(total / len(positions))
        estimated_positions.append(centre)

        # determine estimated chirp positions (equally spaced)
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

        # estimated starting position is the last in the list, plus the chirp length and time between chirps
        position = estimated_positions[-1] + time_between * fs

    return position

# noiseSigma is a tuple of (noiseSigma.real, noiseSignal.imag)
def return_llrs(receivedSymbols, channelEstimates, noiseSigma):
    varianceReal = noiseSigma[0]
    varianceImag = noiseSigma[1]
    channelEstimateMagnitudes = np.abs(channelEstimates) ** 2
    receivedSymbolsReal = receivedSymbols.real
    receivedSymbolsImag = receivedSymbols.imag

    llrsFirstBit = channelEstimateMagnitudes * receivedSymbolsImag / varianceImag # Right now, constellations not normalized so have to add in sqrt(2) later
    llrsSecondBit = channelEstimateMagnitudes * receivedSymbolsReal / varianceReal
    

    llrs = np.empty((llrsFirstBit.size + llrsSecondBit.size,), dtype=llrsFirstBit.dtype)
    llrs[0::2] = llrsFirstBit
    llrs[1::2] = llrsSecondBit
    return llrs


def demapping(qpsk, demappingTable):
    """Demaps from demodulated constellation to original bit sequence"""

    # array of possible constellation points
    constellation = np.array([x for x in demappingTable.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(qpsk.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demappingTable[C] for C in hardDecision]), hardDecision
