import torch
import torch.nn as nn
import numpy as np

from unoptimized.convert import convert_to_unoptimized
from torchvision.models import resnet18, ResNet18_Weights

from unoptimized.modules.linear import UnoptimizedLinear
from unoptimized.modules.conv import UnoptimizedConv2d

if __name__ == '__main__':

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = model.to("cuda:0")
    input = torch.rand((32, 3, 224, 224)).to("cuda:0")
    output1 = model(input)

    unoptimized_linear = UnoptimizedLinear(128, 128, bias=False)
    print(unoptimized_linear)