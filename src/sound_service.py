import numpy as np
import wave


def create_sound(filename):
    res = {
        "file": filename,
        "wave": wave.open(filename, "rb"),
        "data": None
    }
    return res


def calculate_weighted_segment(old_movement, new_movement, nframes, data):
    weight = np.linspace(old_movement, new_movement, nframes * 2)
    return np.multiply(weight, data)


def read_new_segment(wav, nframes):
    return np.fromstring(wav.readframes(nframes), dtype=np.int16)
