"""
Whole model
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import modules


class Net:

    def __init__(self, layers, stacks, in_channels, res_channels, lr=0.002):
        self.net = modules.WaveNet(layers, stacks, in_channels, res_channels)
        self.in_channels = in_channels
        self.receptive_fields = self.net.receptive_fields

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        self._prepare_for_gpu()
    
    
    @staticmethod
    def _loss():
        loss = nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()
        
        return loss
    
    def _optimizer(self):
        return optim.Adam(self.net.parameters(), lr=self.lr)
    
    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))
            self.net = nn.DataParallel(self.net)
        
        if torch.cuda.is_available():
            self.net.cuda()
    
    def train(self, inputs, targets):
        """
        :param inputs: Tensor[batch, channels, timestep]
        :param targets: Tensor[batch, channels, timestep]
        """
        outputs = self.net(inputs)

        loss = self.loss(outputs.view(self.in_channels, -1).transpose(0, 1), targets.long().view(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data[0]
    
    def generate(self, inputs):
        """

        """
        outputs = self.net(inputs)

        return outputs
    
    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = 'wavenet'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))
    
    def load(self, model_dir, step=0):
        """
        """
        print('Loading...')

        model_path = self.get_model_path(model_dir, step)

        self.net.load_state_dict(torch.load(model_path))
    
    def save(self, model_dir, step=0):
        print('Saving...')
        model_path = self.get_model_path(model_dir, step)
        torch.save(self.net.state_dict(), model_path)