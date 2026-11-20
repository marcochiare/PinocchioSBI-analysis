"""
This file contains the neural networks used for
embedding in SBI or for the emulator of the number counts.
"""
import torch
import torch.nn as nn

# ======================================================================================== #
# Helpers
# ======================================================================================== #

def FC_Block_Emu(input_dim, output_dim, p_drop = 0.0):
    return [
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=p_drop)
            ]

def FC_Block_Embed(input_dim, output_dim, dropout = None):

    layers = [nn.Linear(input_dim, output_dim),
              nn.LayerNorm(output_dim),
              nn.ReLU(),
              ]
    
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    
    return layers

# ======================================================================================== #
# Embedding networks
# ======================================================================================== #

class MatrixEmbedding(nn.Module):
    """
    A 2D-CNN+MLP neural network to embed the number counts matrix into a more informative vector.
    Should preserve more information than a pure flattening of the NC matrix.
    """
    def __init__(self, in_shape: tuple, output_dim: int = 32, cnn_init_hidden_chn: int = 16, fc_hidden_chn: int = 128, PoolSize: int = 4):
        super().__init__()

        self.in_shape = in_shape

        self.cnn = nn.Sequential(
                nn.Conv2d(1, cnn_init_hidden_chn, kernel_size=3, padding=1), # (batch, in_hidden_chn, X, X)
                nn.ReLU(),
                nn.Conv2d(cnn_init_hidden_chn, 2*cnn_init_hidden_chn, kernel_size=3, padding=1), # (batch, 2*in_hidden_chn, X, X)
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((PoolSize,PoolSize)), # (batch, 2*in_hidden_chn, PoolSize, PoolSize) 
                nn.Flatten(),   # (batch, 2*in_hidden_chn*PoolSize*PoolSize)
                )

        self.mlp = nn.Sequential(
                nn.Linear(2*cnn_init_hidden_chn*PoolSize*PoolSize, fc_hidden_chn),
                nn.ReLU(),
                nn.Linear(fc_hidden_chn, output_dim),
                )

    def forward(self, x):
        x = x.view(-1, 1, self.in_shape[0], self.in_shape[1])
        return self.mlp(self.cnn(x))

class MatrixEmbeddingSimple(nn.Module):
    """
    A 2D-CNN+MLP neural network to embed the number counts matrix into a more informative vector.
    Should preserve more information than a pure flattening of the NC matrix.

    This version is simpler version.
    """
    def __init__(self, in_shape: tuple, output_dim: int = 32, cnn_init_hidden_chn: int = 16, fc_hidden_chn: int = 128, PoolSize: int = 5):
        super().__init__()

        self.in_shape = in_shape

        self.cnn = nn.Sequential(
                nn.Conv2d(1, cnn_init_hidden_chn, kernel_size=3, padding=1), # (batch, cnn_init_hidden_chn, X, X)
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((PoolSize,PoolSize)), # (batch, cnn_init_hidden_chn, PoolSize, PoolSize) 
                nn.Flatten(),   # (batch, cnn_init_hidden_chn*PoolSize*PoolSize)
                )

        self.mlp = nn.Sequential(
                nn.Linear(cnn_init_hidden_chn*PoolSize*PoolSize, fc_hidden_chn),
                nn.ReLU(),
                nn.Linear(fc_hidden_chn, output_dim),
                )

    def forward(self, x):
        x = x.view(-1, 1, self.in_shape[0], self.in_shape[1])
        return self.mlp(self.cnn(x))

class ConvEmbedding(nn.Module):
    """
    A 1D-CNN+MLP neural network to embed the number counts matrix into a 
    Should preserve more information than a pure flattening of the NC matrix
    """
    def __init__(self, input_dim, output_dim: int = 32, cnn_init_hidden_chn: int = 16, fc_hidden_chn: int = 128, PoolSize: int = 5):
        super().__init__()

        self.cnn = nn.Sequential(
                nn.Conv1d(1, cnn_init_hidden_chn, kernel_size=3, padding=1), # (batch, cnn_init_hidden_chn, X, X)
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((PoolSize,PoolSize)), # (batch, cnn_init_hidden_chn, PoolSize, PoolSize) 
                nn.Flatten(),   # (batch, cnn_init_hidden_chn*PoolSize*PoolSize)
                )

        self.mlp = nn.Sequential(
                nn.Linear(cnn_init_hidden_chn*PoolSize*PoolSize, fc_hidden_chn),
                nn.ReLU(),
                nn.Linear(fc_hidden_chn, output_dim),
                )

    def forward(self, x):
        x = x.view(-1, 1, self.in_shape[0], self.in_shape[1])
        return self.mlp(self.cnn(x))

class FCEmbedding(nn.Module):
    """A MLP neural network to embed the number counts into a more informative vector"""
    def __init__(self, in_channels: int, out_channels: int = 32, hidden_channels: int = 64, n_layers: int = 2, halve: bool = False):
        super().__init__()

        layers = []
        layers.extend(FC_Block_Embed(in_channels, hidden_channels))

        for i in range(n_layers):
            if halve:
                layers.extend(FC_Block_Embed(hidden_channels // 2 ** i, hidden_channels // 2 ** (i + 1)))
            else:
                layers.extend(FC_Block_Embed(hidden_channels, hidden_channels))
        
        if halve:
            layers.append(nn.Linear(hidden_channels // 2 ** n_layers, out_channels))
        else:
            layers.append(nn.Linear(hidden_channels, out_channels))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# ======================================================================================== #
# Emulator networks
# ======================================================================================== #

class FCEmulator(torch.nn.Module):
    """A FC emulator for the number counts"""
    def __init__(self, input_dim, output_dim, hidden_nodes, n_layers,
                 increase: bool = False, p_drop = 0.0):
        super().__init__()

        layers = []
        layers.extend(FC_Block_Emu(input_dim, hidden_nodes, p_drop = p_drop))

        for i in range(n_layers - 1):
            if increase:
                layers.extend(FC_Block_Emu(hidden_nodes * 2 ** i, hidden_nodes * 2 ** (i + 1)))
            else:
                layers.extend(FC_Block_Emu(hidden_nodes, hidden_nodes, p_drop))

        if increase:
            layers.append(nn.Linear(hidden_nodes * 2 ** (n_layers - 1), output_dim))
        else:
            layers.append(nn.Linear(hidden_nodes, output_dim))

        self.fc_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_net(x) 

