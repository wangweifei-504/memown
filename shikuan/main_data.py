import argparse
import os
import random
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from rpad.dataset import prepare_dataset, KPIBatchedWindowDataset, KPIBatchedWindowDataset2, multi_get_data, swat_multi_get_data


##########################################################################################
# Argparse      dest:参数的别名
##################################################### #####################################
def parse_args(verbose=True):
    parser = argparse.ArgumentParser(description='SALAD: KPI Anomaly Detection')

    # Dataset
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument("--data-path", dest='data_path', type=str,
                               default='./data/kpi/series_0.csv',
                               help='The dataset path')
    group_dataset.add_argument("--data-category", dest='data_category',
                               choices=['MSL', 'SMAP', 'Swat', 'machine-1-1', 'machine-1-2',
                                        'machine-1-3', 'machine-1-4', 'machine-1-5',
                                        'machine-1-6', 'machine-1-7', 'machine-1-8',
                                        'machine-2-1', 'machine-2-2', 'machine-2-3',
                                        'machine-2-4', 'machine-2-5', 'machine-2-6',
                                        'machine-2-7', 'machine-2-8', 'machine-2-9',
                                        'machine-3-1', 'machine-3-2', 'machine-3-3',
                                        'machine-3-4', 'machine-3-5', 'machine-3-6',
                                        'machine-3-7', 'machine-3-8', 'machine-3-9',
                                        'machine-3-10', 'machine-3-11'], type=str, default='MSL')

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
    group_model.add_argument("--gen-lr", dest='gen_lr', type=float, default=1e-3)
    group_model.add_argument("--dis-lr", dest='dis_lr', type=float, default=1e-4)
    group_model.add_argument("--pred-lr", dest='pred_lr', type=float, default=1e-4)

    group_model.add_argument("--use-mem", action='store_true')
    group_model.add_argument("--mem-size", type=int, default=1024)
    group_model.add_argument("--use-pred", action='store_true')
    group_model.add_argument("--pred-steps", type=int, default=5)
    group_model.add_argument("--use-birnn", action='store_true')
    group_model.add_argument("--use-birnn-method2", action='store_true')


    group_model.add_argument("--critic", dest='critic_iter', type=int, default=2)
    group_model.add_argument("--rec-weight", dest='rec_weight', type=float, default=1.0)
    group_model.add_argument("--pred-weight", dest='pred_weight', type=float, default=1.0)
    group_model.add_argument("--forwardpred-weight", dest='forwardpred_weight', type=float, default=1.0)
    group_model.add_argument("--backwardpred-weight", dest='backwardpred_weight', type=float, default=1.0)
    group_model.add_argument("--use-regularizer", action='store_true')
    group_model.add_argument("--regularizer-weight", dest='regularizer_weight', type=float, default=0.01)

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
    group_device.add_argument("--ngpu", dest='num_gpu', help="The number of gpu to use", default=3, type=int)
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


if __name__ == '__main__':

    args = parse_args()

    # 当在服务器上运行程序时可以设定设备号，当在集群上调试时则不需要
    torch.cuda.set_device(args.num_gpu)
    # Preparing
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    best_performance = None

    # Reading dataset
    print(f'[INFO] Reading dataset...')

    # Prepare dataset
    if args.data_category == 'Swat':
        (x_train, _), (x_test, y_test) = swat_multi_get_data(args.data_category)
        # print(y_test)
        # print(y_test.shape)
    else:
        (x_train, _), (x_test, y_test) = multi_get_data(args.data_category)
        # print(y_test)
        # print(y_test.shape)

    # Training dataset
    if args.use_birnn_method2:
        train_dataset = KPIBatchedWindowDataset(x_train, label=None, mask=None, window_size=args.window_size,
                                                use_pred=args.use_pred, pred_steps=args.pred_steps, use_birnn=args.use_birnn,
                                                birnn_method2=args.use_birnn_method2)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                  drop_last=True, pin_memory=True)

        test_dataset = KPIBatchedWindowDataset(x_test, label=y_test, mask=None, window_size=args.window_size,
                                               use_pred=args.use_pred, pred_steps=args.pred_steps, use_birnn=args.use_birnn,
                                               birnn_method2=args.use_birnn_method2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False,
                                 pin_memory=True)
    else:
        train_dataset = KPIBatchedWindowDataset2(x_train, label=None, mask=None, window_size=args.window_size,
                                                use_pred=args.use_pred, pred_steps=args.pred_steps,
                                                use_birnn=args.use_birnn,
                                                birnn_method2=args.use_birnn_method2)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                  drop_last=True, pin_memory=True)

        test_dataset = KPIBatchedWindowDataset2(x_test, label=y_test, mask=None, window_size=args.window_size,
                                               use_pred=args.use_pred, pred_steps=args.pred_steps,
                                               use_birnn=args.use_birnn,
                                               birnn_method2=args.use_birnn_method2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)
