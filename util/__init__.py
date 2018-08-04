import pickle
import numpy as np
from .log import log
from .config import config

def as_numeric(s):
    if type(s) == int or type(s) == float or type(s) == bool:
        return s
    try:
        s = int(s)
    except (ValueError, TypeError):
        try:
            s = float(s)
        except (ValueError, TypeError):
            pass
    return s

def save(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def normalize(signal, minimum=None, maximum=None):
    """Normalize a signal to the range 0, 1. Uses the minimum and maximum observed in the data unless explicitly passed."""
    signal = np.array(signal).astype('float')
    if minimum is None:
        minimum = np.min(signal)
    if maximum is None:
        maximum = np.max(signal)
    signal -= minimum
    maximum -= minimum
    signal /= maximum
    signal = np.clip(signal, 0.0, 1.0)
    return signal    

def smooth(signal, size=10, window='blackman'):
    """Apply weighted moving average (aka low-pass filter) via convolution function to a signal with the given window shape and size"""
    """This is going to be faster than highpass_filter"""
    types = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    signal = np.array(signal)
    if size < 3:
        return signal
    s = np.r_[2 * signal[0] - signal[size:1:-1], signal, 2 * signal[-1] - signal[-1:-size:-1]]
    if window == 'flat': # running average
        w = np.ones(size,'d')
    else:
        w = getattr(np, window)(size) # get a series of weights that matches the window and is the correct size 
    y = np.convolve(w / w.sum(), s, mode='same') # convolve the signals
    return y[size - 1:-size + 1]

def upsample(signal, factor):
    """Increase the sampling rate of a signal (by an integer factor), with linear interpolation"""
    assert type(factor) == int and factor > 1
    result = [None] * ((len(signal) - 1) * factor)
    for i, v in enumerate(signal):
        if i == len(signal) - 1:
            result[-1] = v
            break
        v_ = signal[i+1]
        delta = v_ - v
        for j in range(factor):
            f = (i * factor) + j
            result[f] = v + ((delta / factor) * j)
    return result         