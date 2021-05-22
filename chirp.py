import math
import numpy as np

def exponential_chirp(T, f1=60.0, f2=6000.0, window_strength=10.0, fs=44100):
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

def exponential_chirp_rev(T, f1=6000.0, f2=60.0, window_strength=10.0, fs=44100):
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


def exponential_chirp_no_window(T, f1=60.0, f2=6000.0, fs=44100):
    """Produces chirp and returns impulse characteristics"""

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
    """Produces chirp and returns impulse characteristics"""

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

def linear_chirp(T, f1=60.0, f2=6000.0, window_strength=10.0, fs=44100):
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
    t_list = np.linspace(0, T, int(round(T * fs)), False)
    profile = []
    r = f2 / f1

    for t in t_list:
        value = math.sin(2 * math.pi * (f1 + (f2 - f1) * t / T) * t)
        profile.append(value)
    
    # Format
    profile = np.array(profile)

    return profile