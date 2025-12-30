import torch
import torch.nn as nn

#custom l1 distance layer from jupyter nb
class L1Dist(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embedding, validation_embedding):
        return torch.abs(input_embedding - validation_embedding)