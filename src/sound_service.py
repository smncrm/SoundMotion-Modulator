import numpy as np
import wave
from numba import jit

def create_sound(filename):
    res = {
        "file": filename,
        "wave": wave.open(filename, "rb"),
        "data": None
    }
    return res

@jit
def calculate_weighted_segment(old_movement, new_movement, nframes, data):
    weight = np.linspace(old_movement, new_movement, nframes * 2)
    return np.multiply(weight, data)


def read_new_segment(wav, nframes):
    return np.fromstring(wav.readframes(nframes), dtype=np.int16)


def limit_sound(data):
    f = lambda x: np.sign(x) * 31000 if (abs(x) > 31000) else x
    return np.fromiter((f(x) for x in data), data.dtype, count=len(data))

