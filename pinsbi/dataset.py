"""
PINSBI: dataset

This script provides a custom torch Dataset to handle the summary statistics data
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

class SBIDataset(Dataset):
    """
    SBI dataset.
        index: run number,      shape (N,)
        x: summary statistics,  shape (N, nl or nm, nz) or (N, nz*(nl or nm))
        theta: parameters,      shape (N, npars)
    __getitem__ returns (theta, x) at a given idx.
    """
    def __init__(self, indices: np.ndarray, pars: np.ndarray, dataset: np.ndarray):
        assert pars.shape[0] == dataset.shape[0], f'Dimensions do not match. Found {pars.shape[0]} (pars) and {dataset.shape[0] (dataset)}'
        assert pars.shape[0] == indices.shape[0], f'Dimensions do not match. Found {pars.shape[0]} (pars) and {indices.shape[0] (indices)}'

        self.index = torch.tensor(indices, dtype=torch.int)
        self.theta = torch.tensor(pars, dtype=torch.float32)
        self.x = torch.tensor(dataset, dtype=torch.float32)

    def __len__(self):
        return len(self.theta)
    
    def npars(self):
        return self.theta.shape[1]
    
    def __getitem__(self, idx):
        try:
            return self.theta[idx], self.x_pca[idx]
        except AttributeError:
            return self.theta[idx], self.x[idx]

    def to(self, device):
        self.index = self.index.to(device)
        self.theta = self.theta.to(device)
        self.x = self.x.to(device)
        try:
            self.x_pca = self.x_pca.to(device)
        except:
            pass

        return self

    def pca(self, ncomp, verb: bool = True):
        """
        Run the Principal Component analysis for the features vector

        A new attribute SBIDataset.x_pca is then defined and should be used in place
        of SBIDataset.x
        """
        pca = PCA(n_components=ncomp)
        x_pca = pca.fit_transform(self.x)

        if verb:
            print(f'PCA done. Total components: {pca.n_components}, explained variance ratio {pca.explained_variance_ratio_.sum()}')
    
        self.x_pca = torch.tensor(x_pca, dtype=torch.float32)

    def split(self, test_frac: float, seed: int = 883, verb: bool = True):
        """
        Split the dataset in two (training and test) given the test dataset fraction.
        Returns an instance of SBIDataset for each dataset (order: training, test)
        """
        assert test_frac > 0, 'Test fraction cannot be less or equal than zero'
        assert test_frac < 1, 'Test fraction must be smaller than one'
        
        train_frac = 1. - test_frac
        n = len(self)

        self.train_frac = train_frac
        self.test_frac = test_frac

        n_train = int(n * train_frac)
        if verb:
            print('Splitting the dataset')
            print(f'\ttrain ={n_train:6d} ({n_train/n*100:.2f}%)')
            print(f'\ttest  ={n-n_train:6d} ({(n-n_train)/n*100:.2f}%)')
            
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        def _subset(idx):
            dataset = SBIDataset(self.index.numpy()[idx], self.theta.numpy()[idx], self.x.numpy()[idx])
            try:
                dataset.x_pca = self.x_pca[idx]
            except:
                pass
            return dataset

        return (
            _subset(indices[:n_train]),
            _subset(indices[n_train:])
        )

    def split3(self, val_frac: float, test_frac: float, seed: int = 883, verb: bool = True):
        """
        Split the dataset in three (training, validation and test) given the validation and test dataset fractions.
        Returns an instance of SBIDataset for each dataset (order: training, validation, test)
        """
        assert val_frac > 0, 'Val fraction cannot be less or equal than zero'
        assert val_frac < 1, 'Val fraction must be smaller than one'
        assert test_frac > 0, 'Test fraction cannot be less or equal than zero'
        assert test_frac < 1, 'Test fraction must be smaller than one'
        
        train_frac = 1. - test_frac - val_frac
        n = len(self)

        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        if verb:
            print('Splitting the dataset')
            print(f'\ttrain ={n_train:6d} ({n_train/n*100:.2f}%)')
            print(f'\tval   ={n_val:6d} ({(n_val)/n*100:.2f}%)')
            print(f'\ttest  ={n-n_train-n_val:6d} ({(n-n_train-n_val)/n*100:.2f}%)')
            
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        def _subset(idx):
            dataset = SBIDataset(self.index.numpy()[idx], self.theta.numpy()[idx], self.x.numpy()[idx])
            try:
                dataset.x_pca = self.x_pca[idx]
            except AttributeError:
                pass
            return dataset

        return (
            _subset(indices[:n_train]),
            _subset(indices[n_train:n_train+n_val]),
            _subset(indices[n_train+n_val:])
        )


class EmuDataset(Dataset):
    """
    Emulator dataset.
        index: run number,           shape (N,)
        x: inputs  = parameters,     shape (N, npars)
        y: outputs = number counts,  shape (N, nout) or (N, *out_shape)
    __getitem__ returns (y, x) at a given idx.
    """
 
    def __init__(self, indices: np.ndarray, pars: np.ndarray, counts: np.ndarray):
        indices, pars, counts = np.asarray(indices), np.asarray(pars), np.asarray(counts)
        assert pars.ndim == 2, f'`pars` must be 2D (N, npars), found shape {pars.shape}'
        assert indices.shape[0] == pars.shape[0] == counts.shape[0], (
            f'Dimensions do not match. Found {indices.shape[0]} (indices), '
            f'{pars.shape[0]} (pars) and {counts.shape[0]} (counts)')
 
        self.index = torch.tensor(indices, dtype=torch.long)
        self.x = torch.tensor(pars, dtype=torch.float32)
        self.y = torch.tensor(counts, dtype=torch.float32)
 
    def __len__(self):
        """Total elements in the dataset (N)"""
        return len(self.x)
 
    def npars(self):
        """Number of input parameters"""
        return self.x.shape[1]
 
    def nout(self):
        """Number of outputs (flattened)"""
        return int(np.prod(self.y.shape[1:]))
    
    def shapeout(self):
        """Shape of the output (matrix or flattened)"""
        return tuple(self.y.shape[1:])

    def __repr__(self):
        return (f'EmuDataset(N={len(self)}, npars={self.npars()}, nout={self.nout()}, '
                f'x={tuple(self.x.shape)}, y={tuple(self.y.shape)}, device={self.x.device})')

    def __getitem__(self, idx):
        return self.y[idx], self.x[idx]
 
    def to(self, device):
        self.index = self.index.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self
 
    def stats(self):
        """Per-feature mean/std of inputs and outputs. Compute on the TRAINING split only."""
        return {'x_mean': self.x.mean(0), 'x_std': self.x.std(0),
                'y_mean': self.y.mean(0), 'y_std': self.y.std(0)}
 
    def _subset(self, sel):
        return EmuDataset(self.index.cpu().numpy()[sel],
                          self.x.cpu().numpy()[sel],
                          self.y.cpu().numpy()[sel])
 
    def split3(self, val_frac: float, test_frac: float, seed: int = 883, verb: bool = True):
        """
        Split the dataset in three (training, validation, test) given the validation and
        test fractions. Returns an EmuDataset for each (order: training, validation, test).
        """
        assert 0 < val_frac < 1, 'Val fraction must be in (0, 1)'
        assert 0 < test_frac < 1, 'Test fraction must be in (0, 1)'
        assert val_frac + test_frac < 1, 'Val + test fractions must be smaller than one'
 
        train_frac = 1. - test_frac - val_frac
        n = len(self)

        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        if verb:
            print('Splitting the dataset')
            print(f'\ttrain ={n_train:7d} ({n_train / n * 100:.2f}%)')
            print(f'\tval   ={n_val:7d} ({n_val / n * 100:.2f}%)')
            print(f'\ttest  ={n - n_train - n_val:7d} ({(n - n_train - n_val) / n * 100:.2f}%)')
 
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        return (
            self._subset(indices[:n_train]),
            self._subset(indices[n_train:n_train+n_val]),
            self._subset(indices[n_train+n_val:])
        )

