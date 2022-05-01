import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import warnings
from matplotlib.gridspec import GridSpec
# import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from rpad.dataset import prepare_dataset, KPIBatchedWindowDataset
from rpad.model import RPAD


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


##########################################################################################
# Argparse
##################################################### #####################################
def parse_args(verbose=True):
    parser = argparse.ArgumentParser(description='SALAD: KPI Anomaly Detection')

    # Dataset
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument("--data-path", dest='data_path', type=str,
                               default='./data/kpi/series_0.csv',
                               help='The dataset path')
    group_dataset.add_argument("--category", dest='data_category', choices=['kpi', 'nab', 'yahoo'], type=str,
                               default='kpi')
    group_dataset.add_argument("--split", dest='train_val_test_split', type=tuple, default=(5, 2, 3),
                               help='The ratio of train, validation, test dataset')
    group_dataset.add_argument("--filling", dest='filling_method', choices=['zero', 'prev'], default='prev')
    group_dataset.add_argument("--standardize", dest='standardization_method',
                               choices=['standrad', 'minmax', 'negpos1'],
                               default='negpos1')

    # Model
    group_model = parser.add_argument_group('Model')
    group_model.add_argument("--print-model", dest='print_model', action='store_true')
    group_model.add_argument("--var", dest='variant', type=str, choices=['conv', 'dense', 'test'], default='conv')
    group_model.add_argument("--window", dest='window_size', type=int, default=128)
    group_model.add_argument("--hidden", dest='hidden_size', type=int, default=100)
    group_model.add_argument("--latent", dest='latent_size', type=int, default=16)
    group_model.add_argument("--gen-lr", dest='`gen_lr', type=float, default=1e-3)
    group_model.add_argument("--dis-lr", dest='dis_lr', type=float, default=1e-4)
    group_model.add_argument("--use-mem", action='store_true')
    group_model.add_argument("--mem-size", type=int, default=1024)
    group_model.add_argument("--use-pred", action='store_true')

    group_model.add_argument("--critic", dest='critic_iter', type=int, default=2)
    group_model.add_argument("--rec-weight", dest='rec_weight', type=float, default=1.0)
    group_model.add_argument("--contras", dest='use_contrastive', action='store_true')
    # group_model.add_argument("--itimp", dest='in_train_imputation', action='store_true')
    group_model.add_argument("--margin", dest='contrastive_margin', type=float, default=1.0)

    # Save and load
    group_save_load = parser.add_argument_group('Save and Load')
    group_save_load.add_argument("--resume", dest='resume', action='store_true')
    group_save_load.add_argument("--load-path", type=str, default=None)
    group_save_load.add_argument("--save-path", dest='save_path', type=str, default='./cache/uncategorized/')
    group_save_load.add_argument("--interval", dest='save_interval', type=int, default=10)

    # Devices
    group_device = parser.add_argument_group('Device')
    group_device.add_argument("--ngpu", dest='num_gpu', help="The number of gpu to use", default=1, type=int)
    group_device.add_argument("--seed", dest='seed', type=int, default=2019, help="The random seed")

    # Training
    group_training = parser.add_argument_group('Training')
    group_training.add_argument("--epochs", dest="epochs", type=int, default=150, help="The number of epochs to run")
    group_training.add_argument("--batch", dest="batch_size", type=int, default=512, help="The batch size")
    group_training.add_argument("--label-portion", dest="label_portion", type=float, default=0.0,
                                help='The portion of labels used in training')

    # Detection
    group_detection = parser.add_argument_group('Detection')
    group_detection.add_argument("--delay", dest="delay", type=int, default=None, help='The delay of tolerance')
    group_detection.add_argument("--threshold", type=float, default=None,
                                 help='The threshold for determining anomalies')

    args_parsed = parser.parse_args()

    if verbose:
        message = ''
        message += '-------------------------------- Args ------------------------------\n'
        for k, v in sorted(vars(args_parsed).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '-------------------------------- End ----------------------------------'
        print(message)

    return args_parsed


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    # Preparing
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Reading dataset
    print(f'[INFO] Reading dataset...')
    train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m = prepare_dataset(
        args.data_path, args.data_category, args.train_val_test_split,
        1, args.standardization_method, args.filling_method)

    # Training dataset
    train_dataset = KPIBatchedWindowDataset(train_x, label=train_y, mask=train_m, window_size=args.window_size,
                                            use_pred=args.use_pred)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                              drop_last=True, pin_memory=True)

    # Validation dataset
    val_dataset = KPIBatchedWindowDataset(val_x, label=val_y, mask=val_m, window_size=args.window_size,
                                          use_pred=args.use_pred)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True,
                            pin_memory=True)

    test_dataset = KPIBatchedWindowDataset(test_x, label=test_y, mask=test_m, window_size=args.window_size,
                                           use_pred=args.use_pred)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False,
                             pin_memory=True)

    model = RPAD(window_size=args.window_size, hidden_size=args.hidden_size, latent_size=args.hidden_size,
                 batch_size=args.batch_size, variant=args.variant, use_mem=args.use_mem, mem_size=args.mem_size)
    model = model.cuda()

    if args.load_path is None:
        warnings.warn('Using the model without training!')
    else:
        model.load_state_dict(torch.load(args.load_path))

    criterion = nn.MSELoss(reduction='none')

    model.eval()
    reconstructions = []
    observations = []
    scores = []
    labels = []
    with torch.no_grad():
        for i, (x, y, *tails) in enumerate(tqdm(train_loader)):
            x = x.cuda(non_blocking=True)
            x_rec = model(x)

            reconstructions.append(x_rec[:, -1].cpu().numpy())
            observations.append(x[:, -1].cpu().numpy())
            scores.append(criterion(x_rec, x)[:, -1].cpu().numpy())
            labels.append(y[:, -1].numpy())
    reconstructions = np.concatenate(reconstructions, axis=0)
    observations = np.concatenate(observations, axis=0)
    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)

    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    old_len = len(scores)
    scores = smooth(scores)
    if len(scores) < old_len:
        scores = np.concatenate([np.zeros((old_len - len(scores),)), scores])

    if args.load_path is None:
        reconstructions = observations + np.random.randn(*reconstructions.shape)
        reconstructions[labels == 1] += np.random.randn(*reconstructions[labels == 1].shape) * 1.5

    first_anomaly_idx = np.arange(len(labels))[labels == 1][0]
    disp_range = 400

    anomaly_idx = labels[first_anomaly_idx: first_anomaly_idx + disp_range]
    anomaly_idx = np.arange(disp_range)[anomaly_idx == 1]

    print('[INFO] First index:', first_anomaly_idx)

    with plt.style.context(['science', 'grid']):
        grids = GridSpec(ncols=1, nrows=8)
        fig = plt.figure(figsize=(12, 5))

        axes = []

        ax = plt.subplot(grids[:7, :])
        ax.plot(np.arange(disp_range), observations[first_anomaly_idx: first_anomaly_idx + disp_range], color='k',
                label='Original')
        ax.plot(np.arange(disp_range), reconstructions[first_anomaly_idx: first_anomaly_idx + disp_range], 'k--',
                label='Reconstruction')
        ax.scatter(anomaly_idx, observations[first_anomaly_idx: first_anomaly_idx + disp_range][anomaly_idx], s=5,
                   c='red', marker='x')
        ax.legend(fontsize=16, ncol=2, loc='lower right')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        axes.append(ax)

        ax = plt.subplot(grids[-1, :])
        im = ax.imshow(scores[np.newaxis, first_anomaly_idx: first_anomaly_idx + disp_range],
                       cmap='bwr', aspect='auto')
        ax.yaxis.set_visible(False)
        axes.append(ax)

        fig.colorbar(im, ax=axes)
        # fig.tight_layout()
        fig.savefig('./data/ano_spectrogram.pdf')
        plt.show()
