import os
import pandas as pd
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
import torch as tf
from collections import Counter
from datetime import datetime
import math
from sklearn.metrics import confusion_matrix






class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout 
    layers right after a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, 
                 drop=None, bn=True,
                 activ=lambda: nn.PReLU()):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


class Classifier_dh(nn.Module):
    def __init__(self, raw_ni, raw_sz, no, drop=0.5):
        super().__init__()
        self.conv1 = SepConv1d(raw_ni, 32, 8, 2, 3, drop=drop)
        self.conv2 = SepConv1d(32, 32, 3, 1, 1, drop=drop)
        self.conv3 = SepConv1d(32, 64, 8, 4, 2, drop=drop)
        self.conv4 = SepConv1d(64, 64, 3, 1, 1, drop=drop)
        self.conv5 = SepConv1d(64, 128, 8, 4, 2, drop=drop)
        self.conv6 = SepConv1d(128, 128, 3, 1, 1, drop=drop)
        self.conv7 = SepConv1d(128, 256, 8, 4, 2)
        self.flatten = Flatten()
        self.fc1 = nn.Sequential(nn.Dropout(drop), nn.Linear(raw_sz*2, 64), nn.PReLU(), nn.BatchNorm1d(64))
        self.fc2 = nn.Sequential(nn.Dropout(drop), nn.Linear(64, 64), nn.PReLU(), nn.BatchNorm1d(64))
        self.out = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

    def forward(self, t_raw):
#         
        x = self.conv1(t_raw)
#         
        x = self.conv2(x)
#         
        x = self.conv3(x)
#         
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
#        
        x = self.flatten(x)
#         
        x = self.fc1(x)
#         
        x = self.fc2(x)
#         
        x = self.out(x)
        return x
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        

