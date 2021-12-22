import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, frameskip, action_space):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=frameskip, out_channels=32, kernel_size=(8, 8), stride=(4, 4))
        dim_size = conv_output_dim(input_size=84, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))
        dim_size = conv_output_dim(input_size=dim_size, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        dim_size = conv_output_dim(input_size=dim_size, kernel_size=3, stride=1)
        self.lin1 = nn.Linear(in_features=dim_size, out_features=512)
        self.output = nn.Linear(in_features=512, out_features=action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin1(x))
        x = F.softmax(self.output(x))
        return x


def conv_output_dim(input_size, kernel_size, stride=1, padding=0):
    """
    Returns the height or width of the input after a convolution.
    :param input_size: The size (default should be 84 for atari)
    :param kernel_size: The size of the kernel used for convolution
    :param stride: The stride of the convolution
    :param padding: Padding on the convolution
    :return: The size of the dimension after the convolution
    """
    size = int((input_size + 2 * padding - (kernel_size - 1) - 1) / stride)
    return size
