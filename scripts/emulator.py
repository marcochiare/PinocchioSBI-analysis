"""
This file is used to tune and train an emulator of the number counts.
Given a set of cosmological and scaling relation parameters, it produces
the corresponding number counts in (nz, nl) bins (as a row-major flattened vector)
bypassing the computation of the NC from theory.

Accelerate the computation of likelihood-based chains without sacrificing accuracy while
increasing drastically the speed!

Train the neural network on NC from real simulations to obtain a true simulation-based
emulator!

NB: the neural network ignores the physical values of the edges of each bin, it only produces the
values in the bins it sees!
"""
import argparse
import numpy as np
import optuna
import os
import psutil
import time
import torch
import torch.nn as nn
import torch.optim as optim

from pinsbi.dataset import EmuDataset
from pinsbi.neuralnets import FCEmulator
from pinsbi.utils import load_emudataset

from datetime import datetime, timedelta
from functools import partial
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse
from torch.nn import Sequential, Linear, ReLU, ModuleList, BatchNorm1d, Dropout
from torch.utils.data import DataLoader

# ======================================================================================== #
# Defaults
# ======================================================================================== #

# Which number counts probe to read
_DEF_PROBE          = 'counts_lmb_z'
# Flatten the number counts matrix (i.e. neural network output)
FLATTEN             = True
# Type of output normalisation, if any
_DEF_TYPENORM       = 'log'

# Training architecture
_DEF_HIDDNODES      = 256
_DEF_NLAYERS        = 2
_DEF_DROPOUT        = 0.2
_DEF_INCREASE       = False

# Training HPs
_DEF_LR             = 1e-3
_DEF_BATCHSIZE      = 64
_DEF_DECAY          = 1e-4
_DEF_EPS            = 1e-9
_DEF_BETA1          = 0.9
_DEF_BETA2          = 0.999
_DEF_SCHED_PAT      = 10
_DEF_SCHED_FACT     = 0.1
_DEF_NEPOCHS        = 100

# Splitting
_DEF_VALFRAC        = 0.1
_DEF_TESTFRAC       = 0.1
_DEF_SEED           = 883

# Tuning ranges/values
TUNING_NTRIALS      = 4
TUNING_NSTARTUP     = 20
TUNING_MINHIDD      = 8
TUNING_MAXHIDD      = 512
TUNING_HIDDSTEP     = 8
TUNING_MAXLAYERS    = 6
TUNING_MAXDROPOUT   = 0.5
TUNING_LR           = [1e-6, 1e-3]
TUNING_DECAY        = [1e-7, 1e-4]
TUNING_EPS          = [1e-9, 1e-6]
TUNING_BETA1        = [0.85, 0.999]
TUNING_BETA2        = [0.9, 0.9999]
TUNING_SCHED_FACT   = [0.1, 0.4]
TUNING_ES_PATIENCE  = 50

TO_REPORT           = 'last'
_DEF_STUDYNAME      = 'emulator_tuning'

# Normalisation of the inputs
bounds = {
    "Om0":          (0.1, 0.5),
    "sigma8":       (0.60, 1.20),
    "h":            (0.60, 0.80),
    "log10M_1":     (10.5, 11.5),
    "alpha":        (0.1, 1.0),
    "epsi":         (1.0, 1.5),
    "sigm_intr":    (0.1, 0.8),
    "log10M_min":   (9.5, 10.5),
}

# ======================================================================================== #
# Utils
# ======================================================================================== #

def printlog(text: str, flush=False):
    now = datetime.now().strftime('%H:%M:%S')
    print(f'\033[92m[EMULATOR: {now}]>\033[0m ' + text, flush=flush)

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        printlog(f'Created directory: {path}')

def count_params(model):
    """
    Counts the number of trainable parameters for the given model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage(device):
    """
    Returns the CUDA/CPU memory used in MiB
    """
    if device.type.startswith('cuda'):
        mem = torch.cuda.memory_allocated(device) / 1024**2
    else:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024**2

    return mem

class EarlyStopper:

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# ======================================================================================== #
# Training helpers
# ======================================================================================== #

def train(args, dataset: EmuDataset, path_to_weights: str = None):

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')
    printlog(f'Found device {device}')

    input_dim = dataset.npars()
    output_dim = dataset.nout() # always flattened

    # Load the model
    model = FCEmulator(input_dim = input_dim, output_dim = output_dim, hidden_nodes = args.hidden_nodes,
                       n_layers = args.nlayers, increase = args.increase_nodes, p_drop = args.dropout)
    printlog('Model loaded')
    print(model)

    if path_to_weights is not None:
        state = torch.load(path_to_weights, weights_only=True, map_location=device)
        model.load_state_dict(state['model'], True)
        printlog(f'Loaded weights from {path_to_weights}')

    model.to(device)
    printlog(f'Number of parameters: {count_params(model):.3e}', flush=True)

    # Split the dataset
    train_dataset, val_dataset, _ = dataset.split3(val_frac = args.val_frac, test_frac = args.test_frac)
    
    # Load training data
    train_loader = DataLoader(train_dataset,
                              batch_size = args.batchsize,
                              shuffle = True,
                              )

    # Load validation data
    val_loader = DataLoader(val_dataset,
                            batch_size = args.batchsize,
                            shuffle = True,
                            )
    
    # Define the loss function
    criterion = nn.MSELoss()
    criterion.to(device)

    # Define the optimizer and scheduler   
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, eps=args.eps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=_DEF_SCHED_PAT, factor=args.scheduler_factor)

    # Saved states directory
    dirname = f'{args.directory}/torch/'
    create_dir(dirname)

    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d_%H:%M')
    args.checkpt_directory = f'{dirname}/emulator_{formatted_time}'
    create_dir(args.checkpt_directory)

    # Training loop
    printlog('Starting the training')

    losses = []
    val_losses = []
    min_loss = None
    min_epoch = None
    best_state_path = None

    for epoch in range(args.num_epochs):

        # Train
        train_loss = train_epoch(epoch, train_loader, model, criterion,
                                 optimizer, device, args)
        if not args.no_scheduler:
            scheduler.step(train_loss)
        losses.append(train_loss)

        # Validate
        val_loss = validate_epoch(val_loader, model, criterion,
                                  device, args)
        val_losses.append(val_loss)

        # Save epoch state
        if min_loss is None or val_loss < min_loss:
            min_loss = val_loss
            min_epoch = epoch + 1

            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
                'loss':losses,
                'validation_loss':val_losses,
            }

            state_file = f'{args.checkpt_directory}/BestModel_hidd{args.hidden_nodes}_layers{args.nlayers}_epoch{epoch+1}.pt'
            torch.save(state, state_file)
            best_state_path = state_file
            del state

        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
        epoch_msg = f"({formatted_time}), Epoch [{epoch+1:>3}/{args.num_epochs}], Loss: {losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}" 
        print(epoch_msg, flush=True)

    printlog(f'Training done. Minimum validation loss {min_loss:.4f} at epoch {min_epoch}')

    return best_state_path

def train_epoch(epoch: int, loader: DataLoader, model: nn.Module, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, args):
    """
    Train the model for an epoch
    """
    model.train()

    epoch_loss = torch.zeros(1, dtype=torch.float64, device=device)

    for i, (target, x) in enumerate(loader):
        batch = epoch * len(loader) + i + 1 

        target = target.to(device)
        x = x.to(device)
        output = model(x)

        max_mem = get_memory_usage(device)

        # show what is reading/outputting
        if batch <=3:
            print(f'# Running batch: {batch}', flush=True)
            print(f'  Batch size:     {x.shape[0]}', flush=True)
            print(f'  Input  shape:   {x.shape}, size: {x.element_size() * x.nelement() / 1024**2:.2f} MiB', flush=True)
            print(f'  Target shape:   {target.shape}', flush=True)
            print(f'  Output shape:   {output.shape}, size: {output.element_size() * output.nelement() / 1024:.2f} KiB', flush=True)
            print(f'  Max memory for batch: {max_mem:.2f} MiB', flush=True)

        loss = criterion(output, target)
        epoch_loss += loss.detach()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    epoch_loss /= len(loader)

    return epoch_loss.item()

def validate_epoch(loader: DataLoader, model: nn.Module, 
                   criterion: nn.Module, device: torch.device, args):
    """
    Validate the model for an epoch
    """
    model.eval()

    epoch_loss = torch.zeros(1, dtype=torch.float64, device=device)

    with torch.no_grad():
        for target, x in loader:
            target = target.to(device)
            x = x.to(device)
            output = model(x)

            loss = criterion(output, target)
            epoch_loss += loss.detach()

    epoch_loss /= len(loader)

    return epoch_loss.item()

# ======================================================================================== #
# Tuning helpers
# ======================================================================================== #

def suggest_values(trial, args):
    
    # Match the names
    values_dict = {}

    # Architecture
    values_dict['hidden_nodes'] = trial.suggest_int('hidden_nodes', TUNING_MINHIDD, TUNING_MAXHIDD, step=TUNING_HIDDSTEP)
    values_dict['nlayers'] = trial.suggest_int('nlayers', 1, TUNING_MAXLAYERS)
    values_dict['dropout'] = trial.suggest_float('dropout', 0., TUNING_MAXDROPOUT)
    values_dict['increase_nodes'] = trial.suggest_categorical('increase_nodes', [True, False])

    # Optimisation
    values_dict['learning_rate'] = trial.suggest_float('learning_rate', *TUNING_LR, log=True)
    values_dict['weight_decay'] = trial.suggest_float('weight_decay', *TUNING_DECAY, log=True)
    values_dict['eps'] = trial.suggest_float('eps', *TUNING_EPS, log=True)
    values_dict['beta1'] = trial.suggest_float('beta1', *TUNING_BETA1, log=False)
    values_dict['beta2'] = trial.suggest_float('beta2', *TUNING_BETA2, log=False)
    if not args.no_scheduler:
        values_dict['scheduler_factor'] = trial.suggest_float('scheduler_factor', *TUNING_SCHED_FACT, log=False)
    else:
        values_dict['scheduler_factor'] = None # dummy

    printlog(f'Starting configuration (trial {trial.number})', flush=True)
    msg = '; '.join(f'{key}={val}' for key, val in values_dict.items())
    printlog(msg, flush=True)

    return values_dict

def objective(trial, dataset, args):

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    input_dim = dataset.npars()
    output_dim = dataset.nout() # always flattened

    # Get the trial HPs
    suggested_values = suggest_values(trial, args)

    # Load the model
    model = FCEmulator(input_dim = input_dim, output_dim = output_dim,
                       hidden_nodes = suggested_values['hidden_nodes'], 
                       n_layers = suggested_values['nlayers'], 
                       increase = suggested_values['increase_nodes'],
                       p_drop = suggested_values['dropout'],
                       )

    model.to(device)
    printlog(f'Number of parameters: {count_params(model):.3e}', flush=True)

    # Split the dataset
    train_dataset, val_dataset, _ = dataset.split3(val_frac = args.val_frac, test_frac = args.test_frac, verb=(trial.number == 0))
    
    # Load training data
    train_loader = DataLoader(train_dataset,
                              batch_size = args.batchsize,
                              shuffle = True,
                              )

    # Load validation data
    val_loader = DataLoader(val_dataset,
                            batch_size = args.batchsize,
                            shuffle = True,
                            )
    
    # Define the loss function
    criterion = nn.MSELoss()
    criterion.to(device)

    # Define the optimizer and scheduler   
    optimizer = optim.AdamW(model.parameters(), lr=suggested_values['learning_rate'], betas=(suggested_values['beta1'], suggested_values['beta2']),
                            weight_decay=suggested_values['weight_decay'], eps=suggested_values['eps'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=_DEF_SCHED_PAT, factor=suggested_values['scheduler_factor'])

    losses = []
    val_losses = []
    min_loss = None
    min_epoch = None

    early_stopper = EarlyStopper(patience=TUNING_ES_PATIENCE)
    for epoch in range(args.num_epochs):

        # Train
        train_loss = train_epoch(epoch, train_loader, model, criterion,
                                 optimizer, device, args)
        if not args.no_scheduler:
            scheduler.step(train_loss)
        losses.append(train_loss)

        # Validate
        val_loss = validate_epoch(val_loader, model, criterion,
                                  device, args)
        val_losses.append(val_loss)

        if min_loss is None or val_loss < min_loss:
            min_loss = val_loss
            min_epoch = epoch + 1

        trial.report(val_loss, epoch)
    
        # Prune unpromising trials
        if trial.should_prune():
            printlog(f'Trial {trial.number} pruned at epoch {epoch+1}')
            raise optuna.exceptions.TrialPruned()

        # Early stop
        if early_stopper.early_stop(val_loss):
            printlog(f'Trial {trial.number} early stopped at epoch {epoch+1}')
            break

        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
        epoch_msg = f"({formatted_time}), Epoch [{epoch+1:>3}/{args.num_epochs}], Loss: {losses[-1]:.5f}, Validation Loss: {val_losses[-1]:.5f}" 
        print(epoch_msg, flush=True)

    printlog(f'Training done. Minimum validation loss {min_loss:.5f} at epoch {min_epoch}')

    if TO_REPORT == 'min':
        return min_loss
    elif TO_REPORT == 'last':
        return val_losses[-1]
    else:
        raise NameError(f'{TO_REPORT} is not a valid name. Use `min` or `last`.')

def tune(args, dataset: EmuDataset, study_name: str = 'emulator_tuning'):

    # Start optuna study
    dirname = f'{args.directory}/optuna/'
    create_dir(dirname)
    storage = f"sqlite:///{dirname}/{study_name}.db"
    
    sampler = optuna.samplers.TPESampler(n_startup_trials = TUNING_NSTARTUP)
    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=True,
                                direction='minimize',
                                sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(),
                                )

    study.optimize(partial(objective, dataset=dataset, args=args), n_trials=TUNING_NTRIALS, gc_after_trial=True, catch=(torch.OutOfMemoryError))
    
    # Print the results
    best_params = study.best_trial.params
    printlog(f"Best hyperparameters: {str(best_params)}")

    printlog('Overwriting the parsed values with these new hyperparameters for training')
    for key in best_params:
        if key in vars(args):
            setattr(args, key, best_params[key])
        else:
            printlog(f'WARNING: best_trial {key} not found in args')

# ======================================================================================== #
# Testing helpers
# ======================================================================================== #

def test(args, dataset: EmuDataset, path_to_weights):

    printlog('Testing the best model')

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')
    printlog(f'Found device {device}')

    input_dim = dataset.npars()
    output_dim = dataset.nout() # always flattened

    _, _, test = dataset.split3(val_frac = args.val_frac, test_frac = args.test_frac, verb=False)

    # Load the model
    model = FCEmulator(input_dim = input_dim, output_dim = output_dim, hidden_nodes = args.hidden_nodes, n_layers = args.nlayers,
                       increase = args.increase_nodes, p_drop = args.dropout)
    state = torch.load(path_to_weights, weights_only=True, map_location=device)
    model.load_state_dict(state['model'], True)
    printlog(f'Loaded weights from {path_to_weights}')

    model.to(device)
    model.eval()

    test_loader = DataLoader(test, batch_size=args.batchsize, shuffle=False)

    all_preds  = []
    all_truths = []

    with torch.no_grad():
        for target, x in test_loader:
            x = x.to(device)
            target = target.to(device)
            output = model(x)
            all_preds.append(output.cpu().numpy())
            all_truths.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds,  axis=0)
    all_truths = np.concatenate(all_truths, axis=0)
    
    save_data = {'predictions': all_preds, 'targets': all_truths}

    # Save the normalised and un-normalised predictions
    if args.normalise_inputs:
        all_preds_denorm = np.zeros_like(all_preds)
        all_truths_denorm = np.zeros_like(all_truths)

        for i, key in enumerate(bounds.keys()):
            all_preds_denorm[:,i] = denormalize_param(all_preds[:,i], key, bounds)
            all_truths_denorm[:,i] = denormalize_param(all_truths[:,i], key, bounds)

        save_data['predictions_norm'] = save_data['predictions']
        save_data['targets_norm'] = save_data['targets']
        save_data['predictions'] = all_preds_denorm
        save_data['targets'] = all_truths_denorm

    out_path = f'{args.checkpt_directory}/test_predictions.npz'
    np.savez(out_path, **save_data)
    printlog(f'Saved predictions and targets to {out_path}')

    # Print some stats
    printlog(f'Global RMSE:     {rmse(all_truths, all_preds):.4e}')
    printlog(f'Global R2 score: {r2(all_truths, all_preds):.4g}')
    for i, (tgt, pred) in enumerate(zip(all_truths.T, all_preds.T)):
        printlog(f'Parameter {i:>3d} - MSE: {mse(tgt,pred):.4e},  RMSE: {rmse(tgt,pred):.4e},  R2: {r2(tgt,pred):.4g} {"(normalised input val.)" if args.normalise_inputs else ""}')

# ================================================================================================ #

if __name__ == '__main__':

    init_time = time.time()

    # ======================================================================================== #
    # Arg parser
    # ======================================================================================== #

    parser = argparse.ArgumentParser(description='Train an emulator of the number counts')

    # Mode
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--study_name', default=_DEF_STUDYNAME)

    # Main setup
    parser.add_argument('--directory', type=str, required=True)
    parser.add_argument('--numbercounts_path', type=str, required=True)
    parser.add_argument('--path_to_weights', type=str, default=None)
    parser.add_argument('--probe', type=str, default=_DEF_PROBE)
    parser.add_argument('--normalise_inputs', action='store_true')
    parser.add_argument('--normalise_outputs', action='store_true')
    parser.add_argument('--type_norm', type=str, default=_DEF_TYPENORM)

    # Splittings
    parser.add_argument('--val_frac', type=float, default=_DEF_VALFRAC)
    parser.add_argument('--test_frac', type=float, default=_DEF_TESTFRAC)
    parser.add_argument('--seed', type=int, default=_DEF_SEED)

    # Architecture
    parser.add_argument('--hidden_nodes', type=int, default=_DEF_HIDDNODES)
    parser.add_argument('--nlayers', type=int, default=_DEF_NLAYERS)
    parser.add_argument('--dropout', type=float, default=_DEF_DROPOUT)
    parser.add_argument('--increase_nodes', action='store_true')

    # Training HPs
    parser.add_argument('--learning_rate', type=float, default=_DEF_LR)
    parser.add_argument('--batchsize', type=int, default=_DEF_BATCHSIZE)
    parser.add_argument('--weight_decay', type=float, default=_DEF_DECAY)
    parser.add_argument('--beta1', type=float, default=_DEF_BETA1)
    parser.add_argument('--beta2', type=float, default=_DEF_BETA2)
    parser.add_argument('--eps', type=float, default=_DEF_EPS)
    parser.add_argument('--num_epochs', type=int, default=_DEF_NEPOCHS)
    parser.add_argument('--scheduler_factor', type=float, default=_DEF_SCHED_FACT)
    parser.add_argument('--no_scheduler', action='store_true')

    args = parser.parse_args()

    # ======================================================================================== #
    # Dataset
    # ======================================================================================== #

    dataset, info = load_emudataset(args.numbercounts_path, bounds.keys(), args.probe,
                                    normalise_in=args.normalise_inputs, normalise_out=args.normalise_outputs,
                                    type_norm=_DEF_TYPENORM, bounds=bounds, flatten=FLATTEN, return_info=True)

    # ======================================================================================== #
    # Tune
    # ======================================================================================== #
    
    if args.tune:
        printlog('Tuning the emulator of the number counts')
        tune(args, dataset, study_name=args.study_name)

    # ======================================================================================== #
    # Train
    # ======================================================================================== #

    printlog('Training the emulator of the number counts')
    best_res = train(args, dataset, path_to_weights=args.path_to_weights)

    # ======================================================================================== #
    # Test
    # ======================================================================================== #
    
    test(args, dataset, best_res)

    final_time = time.time() - init_time
    printlog(f'Finished. Elapsed wall-clock time: {str(timedelta(seconds=int(final_time)))}')

