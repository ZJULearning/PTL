import torch
from torch import nn
import math


class BConvCell(nn.Module):
    """ Init a Batch-related convolutional cell.
        'Progressive transfer learning for person re-identification' by Yu et al.
    Args:
        input_dim: Dimension of the input feature maps
        output_dim: Dimension of the output feature maps
        kernel_size: Kernel size of the convolutional layer, same usage as in 'nn.Conv2d'
        stride: Stride of the convolutional layer, same usage as in 'nn.Conv2d'
        padding: Padding of the convolutional layer, same usage as in 'nn.Conv2d'
    """

    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=0):
        super(BConvCell, self).__init__()
        self.output_dim = output_dim
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.gates = nn.Conv2d(input_dim, 4 * output_dim, kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.latentState = None

    def resetlatentstate(self):
        self.latentState=None

    def forward(self, input_):
        # get batch size and spatial sizes
        batch_size = input_.size(0)
        spatial_size = input_.data.size()[2:]
        # generate empty prev_state, if None is provided
        height = int(math.floor(
            ((list(spatial_size)[0] + 2 * self.padding - (self.kernel_size - 1) - 1) / float(self.stride)) + 1))
        weight = int(
            math.floor((list(spatial_size)[1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride) + 1)
        if self.latentState is None:
            cell_state_size = [batch_size, self.output_dim] + [height, weight]
            prev_states = torch.nn.Parameter(torch.zeros(cell_state_size)).cuda()
            nn.init.normal_(prev_states, std=0.001)
        else:
            prev_states = self.latentState.detach()
        if prev_states.size(0) != batch_size:
            prev_states = prev_states[:batch_size]
        gates = self.gates(input_)
        # chunk across channel dimension
        in_gate, forget_gate, out_gate, cell_gate = gates.chunk(4, 1)
        # apply sigmoid on input gate, forget gate and output gate
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        # apply tanh on cell gate
        cell_gate = torch.tanh(cell_gate)
        # update the latent state
        now_state = (forget_gate * prev_states) + (in_gate * cell_gate)
        # use the latent state to rectify the output feature map
        output = out_gate * torch.tanh(now_state)
        self.latentState = now_state.detach()
        return output