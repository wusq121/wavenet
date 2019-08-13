"""
data load and preprocess
"""
import os
import librosa
import numpy as np

import torch
import torch.utils.data as data


def load_audio(filename, sample_rate=22500, trim=True, trim_frame_length=2048):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)

    if trim:
        audio._ = librosa.effects.trim(audio, frame_length=trim_frame_length)

    return audio


def one_hot_encode(data, channels=256):
    """
    the return of this function is a numpy array shaped as [C(channels), L(timestep)]
    """
    one_hot = np.zeros((channels, data.size), dtype=float)
    one_hot[data.ravel(), np.arange(data.size)] = 1

    return one_hot


def one_hot_decode(data, axis=0):
    decoded = np.argmax(data, axis=axis)

    return decoded


def quantize_encode(audio, quantization=256):
    mu = float(quantization - 1)
    quantization_space = np.linspace(-1, 1, quantization)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu +
                                                                         1)
    quantized = np.digitize(quantized, quantization_space) - 1

    return quantized


def quantize_decode(quantized, quantization=256):
    mu = float(quantization - 1)
    expand = (quantized / quantization) * 2.0 - 1
    waveform = np.sign(expand) * (np.exp(np.abs(expand) * np.log(1 + mu)) -
                                  1) / mu

    return waveform


class Audioset(data.Dataset):
    """
    When get an item in the dataset, the audio is shaped as [C(channel), L(timestep)]
    """
    def __init__(self, data_dir, sample_rate=22500, in_channels=256,
                 trim=True):
        super(Audioset, self).__init__()
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.trim = trim
        self.root_path = data_dir
        self.filename = [x for x in sorted(os.listdir(data_dir))]

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filename[index])
        raw_audio = load_audio(filepath, self.sample_rate, self.trim)
        encode = one_hot_encode(quantize_encode(raw_audio, self.in_channels),
                                self.in_channels)

        return encode

    def __len__(self):
        return len(self.filename)


class DataLoader(data.DataLoader):
    def __init__(self,
                 data_dir,
                 receptive_fields,
                 sample_size=0,
                 sample_rate=22500,
                 in_channels=256,
                 batch_size=1,
                 shuffle=True):
        """
        DataLoader for Network
        :param data_dir: directory of data
        :param receptive_fields: size of receptive fields.
        :param sample_size: number of timesteps to train at once. sample size has to be bigger than receptive fields.
        :param sample_rate: sound sampling rate
        :param in_channels: number of input channels
        :param batch_size:
        :param shuffle:
        """
        dataset = Audioset(data_dir, sample_size, in_channels)

        super(DataLoader, self).__init__(dataset, batch_size, shuffle)

        if sample_rate <= receptive_fields:
            raise Exception(
                "sample_size has to be bigger than receptive_fields")

        self.sample_size = sample_size
        self.receptive_fields = receptive_fields
        self.collate_fn = self._collate_fn

    def calc_sample_size(self, audio):
        return self.sample_size if len(audio[0]) >= self.sample_size else len(
            audio[0])


    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)
    
    def _collate_fn(self, audio):
        audio = np.pad(audio, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')

        if self.sample_size:
            sample_size = self.calc_sample_size(audio)

            while sample_size > self.receptive_fields:
                inputs = audio[:, :sample_size, :]
                targets = audio[:, self.receptive_fields, :]
            
                yield self._variable(inputs), self._variable(one_hot_decode(targets, 2))
            
                audio = audio[:, sample_size - self.receptive_fields:, :]
                sample_size = self.calc_sample_size(audio)
        
        else:
            targets = audio[:, self.receptive_field:, :]
            return self._variable(audio), self._variable(one_hot_decode(targets, 2))
        
