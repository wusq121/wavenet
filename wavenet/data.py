"""
data load and preprocess
"""
import os
import librosa
import numpy as np

import torch
import torch.utils.data as data


def load_audio(filename, sample_rate = 22500, trim=True, trim_frame_length=2048):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)#?

    if trim:
        audio. _ = librosa.effects.trim(audio, frame_length=trim_frame_length)
    
    return audio


def one_hot_encode(data, channels=256):
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data.revel()] = 1

    return one_hot


def one_hot_decode(data, axis=1):
    decoded = np.argmax(data, axis=axis)

    return decoded


def quantize_encode(audio, quantization=256):
    mu = float(quantization - 1)
    quantization_space = np.linspace(-1, 1, quantization)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantization_space) - 1

    return quantized

def quantize_decode(quantized, quantization=256):
    mu = float(quantization - 1)
    expand = (quantized / quantization) * 2.0 - 1
    waveform = np.sign(expand) * (np.exp(np.abs(expand) * np.log(1 + mu)) - 1) / mu

    return waveform


class Audioset(data.Dataset):
    def __init__(self, data_dir, sample_rate=22500, in_channels=256, trim=True):
        super(Audioset, slef).__init__()
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.trim = trim
        self.root_path = data_dir
        self.filename = [x for x in sorted(os.listdir(data_dir))]

    
    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filename[index])
        raw_audio = load_audio(filepath, self.sample_rate, self.trim)
        encode = one_hot_encode(quantize_encode(raw_audio, self.in_channels), self.in_channels)

        return encode

    def __len__(self):
        return len(self.filename)
    

class DataLoader(data.DataLoader):
    def __init__(
        self, data_dir,
        receptive_fields,
        sample_size=0,
        sample_rate=22500,
        in_channels=256,
        batch_size=1,
        shuffle=True
    ):
        """
        DataLoader for Network
        :param data_dir:
        :param receptive_fields:
        
        """
        


