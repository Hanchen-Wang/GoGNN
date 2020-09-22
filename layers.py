import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNEdgeAttribute:
    def __init__(self, in_dim, out_dim, bias=True):
        super(GCNEdgeAttribute, self).__init__()
