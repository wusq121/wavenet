"""
Nural network modules for wavenet

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from exceptions import InputSizeError 

class CausalConv1d(nn.Conv1d):
    """
    Dilated Causal Convolution for WaveNet
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True
    ):
        """
        :param in_channels: number of channels in the input data;
        :param out_channels: number of channels produced by the convolution;
        :param kernel_size: size of the convolving kernel;
        :param stride: stride of the convolution. Default: 1;
        :param dilation: spacing between kernel elements, and determining the left padding. Default: 1.
        :param groups: number of blocked connections from input channels to out channels. Default: 1;
        :param bias: if True, add a learnable bias to the output. Default: 1.
        """
        super(CausalConv1d,self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.left_padding = dilation * (kernel_size - 1)
    
    def forward(self, input):
        x = F.pad(input.unsqueeze(2),(self.left_padding, 0, 0, 0)).squeeze(2)

        return super(CausalConv1d, self).forward(x)
    

class ResidualBlock(nn.Module):
    """Residual block for wavenet"""
    def __init__(self, res_channels, skip_channels, dilation):
        """
        :param res_channels: number of residual channel for input, output;
        :param skip_cahnnels: number of skip for output.
        :param dilation: dilation for the causal conv1d.
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = CausalConv1d(
            res_channels, 
            res_channels, 
            kernel_size=2, 
            dilation=dilation, 
            bias=False
        )
        self.conv_res = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.gate_tanh = nn.Tanh()
        self.gate_sigmoid = nn.Sigmoid()
    

    def forward(self, x, skip_size):
        """
        :param x: input data, size is [N, C, L];
        :param skip_size: the last output size for loss and prediction;
        :return output: output of this residual block;
        :return skip: output as input for skip connections.
        """
        output = self.conv1(x)
        gate_t = self.gate_tanh(output)
        gate_s = self.gate_sigmoid(output)
        gated = gate_t * gate_s

        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        output = output + input_cut

        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]

        return output, skip
    
class ResidualStack(nn.Module):
    """Stack residual blocks by layer and stack size"""

    def __init__(self, layers, stacks, res_channels, skip_channels):
        """
        :param layers: 
        :param stacks:
        :param res_channels:
        :param skip_channels:
        """
        super(ResidualStack, self).__init__()
            
        self.layers = layers
        self.stacks = stacks

        self.res_blocks = self.stack_res_block(res_channels, skip_channels)
        
    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)
        
        if torch.cuda.device_count() > 1:
            block = nn.DataParallel(block)
            
        if torch.cuda.is_available():
                block.cuda()
            
        return block
        
    def build_dilations(self):
        dilations = []

        for s in range(0, self.stacks):
            for l in range(0, self.layers):
                dilations.append(2 ** l)
        
        return dilations
        
    def stack_res_block(self, res_channels, skip_channels):
        """
        Prepare dilated convolution(res) blocks by layer and stack size
        """
        res_blocks = []
        dilations = self.build_dilations()

        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation)
            res_blocks.append(block)
            
        return res_blocks
        
    def forwward(self, x, skip_size):
        """
        :param x: input data
        :param skip_size: The last output size for loss and prediction
        """

        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)
        
        return torch.stack(skip_connections)

class DensNet(nn.Module):
    """
    the last network of wavenet
    :param x: input data(skip_connections)
    """
    def __init__(self, channels):
        """
        :param channels: number of channels for input and output
        """
        super(DensNet, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        output = self.conv1(F.relu(x))
        output = self.conv2(F.relu(output))

        output = F.softmax(output)

        return output
        

class WaveNet(nn.Module):
    """
    The size of timestep (3rd dimension) has to larger than receptive fields.
    :param x: Tensor[batch, channels, timestep]
    """
    def __init__(self, layers, stacks, in_channels, res_channels):
        """
        :param layers: 
        :param stacks:
        :param in_channels: number of channels for input data, skip_channel in same as input channel;
        :param res_channels: number of residual for input, output.
        """
        super(WaveNet, self).__init__()

        self.receptive_fields = self._receptive_field(layers, stacks)
        self.causal = CausalConv1d(in_channels, res_channels, 2, bias=False)
        self.res_stack = ResidualStack(layers, stacks, res_channels, in_channels)
        self.densnet = DensNet(in_channels)
    
    @staticmethod
    def _receptive_filed(layers, stacks):
        layer = [2 ** i for i in range(layers)] * stacks
        num_receptive_field = np.sum(layer)
        
        return int(num_receptive_field)
    
    def calc_out_size(self, x):
        out_size = int(x.size(2)) - self.receptive_fields

        self.check_input_size(x, out_size)

        return out_size
    

    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise InputSizeError(int(x.size(2)), self.receptive_fields, output_size)
    

    def forward(self, x):
        output_size = self.calc_out_size(x)
        output = self.causal(x)
        skip_connections = self.res_stack(output, output_size)
        output = torch.sum(skip_connections, dim=0)
        output = self.densnet(output)
        return output.contiguous()



