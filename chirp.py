import math
import numpy as np
from graphing_utils import *


def exponential_chirp(T, f1=60.0, f2=6000.0, window_strength=10.0, fs=44100):
    """Produces exponential chirp with exponential envelope"""

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


def exponential_chirp_chain(T=1, f1=60.0, f2=6000.0, window_strength=50.0, fs=44100, number_chirps=3, time_between=1):
    """Produces exponential chirp chain with exponential envelope"""

    x = exponential_chirp(T, f1, f2, window_strength, fs)

    x_chain = []

    for i in range(round((len(x) + fs * time_between) * number_chirps)):
        if i % round((len(x) + time_between * fs)) < len(x):
            x_chain.append(x[i % round((len(x) + time_between * fs))])
        else:
            x_chain.append(0.0)

    x_chain = np.array(x_chain)

    return x_chain


def exponential_chirp_rev(T, f1=6000.0, f2=60.0, window_strength=10.0, fs=44100):
    """Produces exponential chirp with exponential envelope reversed"""

    t_list = np.linspace(0, T, int(round(T * fs)), False)
    profile = []
    r = f2/f1

    # Calculate Sine Sweep time domain values
    for t in t_list:
        value = math.sin(2*math.pi*T*f1*((r**(t/T)-1)/(math.log(r, math.e))))*(1-math.e**(-window_strength*t))*(1-math.e**(window_strength*(t-T)))
        profile.append(value)

    # Format
    profile = np.array(profile)
    profile = profile[::-1]

    return profile


def exponential_chirp_no_window(T, f1=60.0, f2=6000.0, fs=44100):
    """Produces exponential chirp"""

    t_list = np.linspace(0, T, int(round(T * fs)), False)
    profile = []
    r = f2/f1

    # Calculate Sine Sweep time domain values
    for t in t_list:
        value = math.sin(2*math.pi*T*f1*((r**(t/T)-1)/(math.log(r, math.e))))
        profile.append(value)

    # Format
    profile = np.array(profile)

    return profile


def exponential_chirp_no_window_rev(T, f1=6000.0, f2=60.0, fs=44100):
    """Produces exponential chirp reversed"""

    t_list = np.linspace(0, T, int(round(T * fs)), False)
    profile = []
    r = f2/f1

    # Calculate Sine Sweep time domain values
    for t in t_list:
        value = math.sin(2*math.pi*T*f1*((r**(t/T)-1)/(math.log(r, math.e))))
        profile.append(value)

    # Format
    profile = np.array(profile)
    profile = profile[::-1]

    return profile


def linear_chirp(T, f1=60.0, f2=6000.0, window_strength=10.0, fs=44100):
    """Produces linear chirp with exponential envelope"""

    t_list = np.linspace(0, T, int(round(T * fs)), False)
    profile = []
    r = f2/f1

    for t in t_list:
        value = math.sin(2 * math.pi * (f1 + (f2 - f1) * t / T) * t) * (1-math.e**(-window_strength*t))*(1-math.e**(window_strength*(t-T)))
        profile.append(value)
    
    # Format
    profile = np.array(profile)

    return profile


def linear_chirp_no_window(T, f1=60.0, f2=6000.0, fs=44100):
    """Produces linear chirp"""

    t_list = np.linspace(0, T, int(round(T * fs)), False)
    profile = []
    r = f2 / f1

    for t in t_list:
        value = math.sin(2 * math.pi * (f1 + (f2 - f1) * t / T) * t)
        profile.append(value)
    
    # Format
    profile = np.array(profile)

    return profile
