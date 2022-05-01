import argparse
import csv
import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from itertools import chain
# import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from rpad.dataset import prepare_dataset, KPIBatchedWindowDataset, KPIBatchedWindowDataset2, multi_get_data, \
    swat_multi_get_data
from rpad.misc import print_blue_info
from rpad.model import RPAD, WeightedPredictionLoss
from rpad.util import calculate_performance, smd_calculate_performance


def setup_seed(seed):
    warnings.warn(f'You have chosen to seed ({seed}) training. This will turn on the CUDNN deterministic setting, '
                  f'which can slow down your training considerably! You may see unexpected behavior when restarting '
                  f'from checkpoints.')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True # 保证每次卷积算法返回结果一样


##########################################################################################
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
    group_training.add_argument("--epochs", dest="epochs", type=int, default=2, help="The number of epochs to run")
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


def train(model, train_loader, test_loader, args):
    # Optimizers
    # optimizer_enc = torch.optim.Adam(model.encoder.parameters(), lr=args.gen_lr, betas=(0.5, 0.999),
    #                                  weight_decay=2.5 * 1e-5)
    # optimizer_dec = torch.optim.Adam(model.decoder.parameters(), lr=args.gen_lr, betas=(0.5, 0.999),
    #                                  weight_decay=2.5 * 1e-5)
    if args.use_mem:
        # chain ??????
        optimizer_autoencoder = optim.Adam(
            chain(model.encoder.parameters(), model.decoder.parameters(), model.memory.parameters()), lr=args.gen_lr,
            betas=(0.5, 0.999), weight_decay=2.5 * 1e-5)
    else:
        optimizer_autoencoder = optim.Adam(
            chain(model.encoder.parameters(), model.decoder.parameters()), lr=args.gen_lr, betas=(0.5, 0.999),
            weight_decay=2.5 * 1e-5)
    optimizer_enc = optim.Adam(
        model.encoder.parameters(), lr=args.gen_lr, betas=(0.5, 0.999), weight_decay=2.5 * 1e-5
    )
    optimizer_ddis = optim.Adam(model.data_discriminator.parameters(), lr=args.dis_lr, betas=(0.5, 0.999),
                                weight_decay=2.5 * 1e-5)
    optimizer_ldis = optim.Adam(model.latent_discriminator.parameters(), lr=args.dis_lr, betas=(0.5, 0.999),
                                weight_decay=2.5 * 1e-5)
    if args.use_pred:
        optimizer_pred = optim.Adam(model.predictor.parameters(), lr=args.pred_lr, betas=(0.5, 0.999),
                                    weight_decay=2.5 * 1e-5)
        # pred Schedulers
        scheduler_pred = torch.optim.lr_scheduler.StepLR(optimizer_pred, step_size=10, gamma=0.75)

    # Schedulers
    scheduler_enc = torch.optim.lr_scheduler.StepLR(optimizer_enc, step_size=10, gamma=0.75)
    scheduler_autoencoder = torch.optim.lr_scheduler.StepLR(optimizer_autoencoder, step_size=10, gamma=0.75)
    scheduler_ddis = torch.optim.lr_scheduler.StepLR(optimizer_ddis, step_size=10, gamma=0.75)
    scheduler_ldis = torch.optim.lr_scheduler.StepLR(optimizer_ldis, step_size=10, gamma=0.75)

    model.train()
    # Train epoch
    for epoch in range(args.epochs):
        # Train batch
        rec_losses = []
        data_gen_losses = []
        data_dis_losses = []
        data_pred_losses = []
        latent_gen_losses = []
        latent_dis_losses = []
        for i, (x, *tails) in enumerate(tqdm(train_loader, desc='EPOCH: [%d/%d]' % (epoch + 1, args.epochs))):
            # no label while training
            if args.use_pred:
                if args.use_birnn:
                    if args.use_birnn_method2:
                        x_next = tails[0]
                        x_next = x_next.permute(0, 2, 1)
                        y_pred_forward = tails[1]
                        y_pred_backward = tails[2]
                        y_pred_forward = y_pred_forward.permute(0, 2, 1)
                        y_pred_backward = y_pred_backward.permute(0, 2, 1)
                        if torch.cuda.is_available():
                            y_pred_forward = y_pred_forward.cuda(non_blocking=True)
                            y_pred_backward = y_pred_backward.cuda(non_blocking=True)
                            y_pred = tuple([y_pred_forward, y_pred_backward])
                            x_next = x_next.cuda(non_blocking=True)
                    else:
                        y_pred_forward = tails[0]
                        y_pred_backward = tails[1]
                        y_pred_forward = y_pred_forward.permute(0, 2, 1)
                        y_pred_backward = y_pred_backward.permute(0, 2, 1)
                        if torch.cuda.is_available():
                            y_pred_forward = y_pred_forward.cuda(non_blocking=True)
                            y_pred_backward = y_pred_backward.cuda(non_blocking=True)
                            y_pred = tuple([y_pred_forward, y_pred_backward])
                else:
                    y_pred = tails[0]
                    y_pred = y_pred.permute(0, 2, 1)

                    if torch.cuda.is_available():
                        y_pred = y_pred.cuda(non_blocking=True)

            x = x.permute(0, 2, 1)
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)

            ##################################################################################
            # Data discrimination
            ##################################################################################
            for k in range(args.critic_iter):
                if args.use_birnn_method2:
                    data_dis_loss = model.data_dis_loss(x, x_next)
                    optimizer_ddis.zero_grad()
                    data_dis_loss.backward()
                    optimizer_ddis.step()

                    data_dis_losses.append(data_dis_loss.item())
                else:
                    data_dis_loss = model.data_dis_loss(x)
                    optimizer_ddis.zero_grad()
                    data_dis_loss.backward()
                    optimizer_ddis.step()

                    data_dis_losses.append(data_dis_loss.item())


            ##################################################################################
            # Data Model loss
            ##################################################################################

            if args.use_pred:

                if args.use_birnn:
                    if args.use_birnn_method2:
                        if args.use_regularizer:
                            data_gen_loss, data_rec_loss, data_pred_loss, z_regularizer = model.data_gen_loss(x,
                                                                                                              x_next=x_next,
                                                                                                              y_pred=y_pred)
                            data_loss = data_gen_loss + args.rec_weight * data_rec_loss + args.forwardpred_weight * \
                                        data_pred_loss[0] + \
                                        args.backwardpred_weight * data_pred_loss[
                                            1] + args.regularizer_weight * z_regularizer
                        else:
                            data_gen_loss, data_rec_loss, data_pred_loss = model.data_gen_loss(x, x_next=x_next,
                                                                                               y_pred=y_pred)
                            data_loss = data_gen_loss + args.rec_weight * data_rec_loss + args.forwardpred_weight * \
                                        data_pred_loss[0] + args.backwardpred_weight * data_pred_loss[1]
                    else:
                        if args.use_regularizer:
                            data_gen_loss, data_rec_loss, data_pred_loss, z_regularizer = model.data_gen_loss(x,
                                                                                                              y_pred=y_pred)
                            data_loss = data_gen_loss + args.rec_weight * data_rec_loss + args.forwardpred_weight * \
                                        data_pred_loss[0] + \
                                        args.backwardpred_weight * data_pred_loss[
                                            1] + args.regularizer_weight * z_regularizer
                        else:
                            data_gen_loss, data_rec_loss, data_pred_loss = model.data_gen_loss(x, y_pred=y_pred)
                            data_loss = data_gen_loss + args.rec_weight * data_rec_loss + args.forwardpred_weight * \
                                        data_pred_loss[0] + args.backwardpred_weight * data_pred_loss[1]


                else:
                    if args.use_regularizer:
                        data_gen_loss, data_rec_loss, data_pred_loss, z_regularizer = model.data_gen_loss(x,
                                                                                                          y_pred=y_pred)
                        data_loss = data_gen_loss + args.rec_weight * data_rec_loss + args.pred_weight * data_pred_loss + args.regularizer_weight * z_regularizer
                    else:
                        data_gen_loss, data_rec_loss, data_pred_loss = model.data_gen_loss(x, y_pred=y_pred)
                        data_loss = data_gen_loss + args.rec_weight * data_rec_loss + args.pred_weight * data_pred_loss

            else:

                if args.use_regularizer:
                    data_gen_loss, data_rec_loss, z_regularizer = model.data_gen_loss(x)
                    data_loss = data_gen_loss + args.rec_weight * data_rec_loss + args.regularizer_weight * z_regularizer
                else:
                    data_gen_loss, data_rec_loss = model.data_gen_loss(x)
                    data_loss = data_gen_loss + args.rec_weight * data_rec_loss
            # wandb.log({'data_gen_loss': data_gen_loss.item(), 'data_rec_loss': data_rec_loss.item(),
            #            'data_loss': data_loss.item()})
            optimizer_autoencoder.zero_grad()
            if args.use_pred:
                optimizer_pred.zero_grad()
            data_loss.backward()
            optimizer_autoencoder.step()
            if args.use_pred:
                optimizer_pred.step()

            data_gen_losses.append(data_gen_loss.item())
            rec_losses.append(data_rec_loss.item())
            if args.use_pred:
                if args.use_birnn:
                    data_pred_losses.append(data_pred_loss[0].item() + data_pred_loss[1].item())
                else:
                    data_pred_losses.append(data_pred_loss.item())

            ##################################################################################
            # Latent discrimination
            ##################################################################################
            if not args.use_mem:
                # Latent discriminator
                for k in range(args.critic_iter):
                    latent_dis_loss = model.latent_dis_loss(x)
                    optimizer_ldis.zero_grad()
                    latent_dis_loss.backward()
                    optimizer_ldis.step()
                    # wandb.log({'latent_dis_loss%d' % k: latent_dis_loss.item()})
                    latent_dis_losses.append(latent_dis_loss.item())

                # Generator
                latent_gen_loss = model.latent_gen_loss(x)
                optimizer_enc.zero_grad()
                latent_gen_loss.backward()
                optimizer_enc.step()
                latent_gen_losses.append(latent_gen_loss.item())
        # Learning rate adjustment
        scheduler_enc.step()
        scheduler_autoencoder.step()
        scheduler_ddis.step()
        scheduler_ldis.step()
        if args.use_pred:
            scheduler_pred.step()

        # Save state dicts
        if (epoch + 1) % args.save_interval == 0:

            torch.save(model.state_dict(), os.path.join(args.save_path, f'model_epoch{epoch + 1}.pth.tar'))

        if epoch >= 49 and (epoch + 1) % 10 == 0:
            print('========== EPOCH %d==========' % (epoch + 1))
            evaluate(model=model, test_loader=test_loader, delay=delay, args=args)
            model.train()

        print('dgen: {} - ddis: {}\nzgen: {} - zdis: {}\nrec: {} - pred: {}'.format(np.mean(data_gen_losses),
                                                                                    np.mean(data_dis_losses),
                                                                                    np.mean(latent_gen_losses),
                                                                                    np.mean(latent_dis_losses),
                                                                                    np.mean(rec_losses),
                                                                                    np.mean(data_pred_losses)))


def evaluate(model, test_loader, delay, args):
    y_pred = []
    y_true = []

    pred_criterion = WeightedPredictionLoss(args.pred_steps, reduction='none')

    model.eval()
    with torch.no_grad():
        for i, (x, *tails) in enumerate(tqdm(test_loader)):

            x = x.permute(0, 2, 1)
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
            if args.use_pred:
                # _, pred_target = tails
                if args.use_birnn:
                    if args.use_birnn_method2:
                        x_next = tails[0]
                        x_next = x_next.permute(0, 2, 1)
                        y = tails[1]
                        pred_target_forward = tails[2].permute(0, 2, 1)
                        pred_target_backward = tails[3].permute(0, 2, 1)
                        if torch.cuda.is_available():
                            x_next = x_next.cuda(non_blocking=True)
                    else:
                        y = tails[0]
                        pred_target_forward = tails[1].permute(0, 2, 1)
                        pred_target_backward = tails[2].permute(0, 2, 1)

                    if torch.cuda.is_available():
                        y = y.cuda(non_blocking=True)
                        pred_target_forward = pred_target_forward.cuda(non_blocking=True)
                        pred_target_backward = pred_target_backward.cuda(non_blocking=True)
                else:
                    y = tails[0]
                    pred_target = tails[1].permute(0, 2, 1)
                    if torch.cuda.is_available():
                        y = y.cuda(non_blocking=True)
                        pred_target = pred_target.cuda(non_blocking=True)

                if args.use_mem:
                    if args.use_birnn_method2:
                        if args.use_regularizer:
                            x_rec, x_pred, z_regularizer = model(x, x_next=x_next, use_mem=args.use_mem,
                                                                 use_reg=args.use_regularizer,
                                                                 use_pred=args.use_pred,
                                                                 use_birnn=args.use_birnn,
                                                                 use_birnn_method2=args.use_birnn_method2)
                        else:
                            x_rec, x_pred = model(x, x_next=x_next, use_mem=args.use_mem,
                                                  use_reg=args.use_regularizer,
                                                  use_pred=args.use_pred,
                                                  use_birnn=args.use_birnn,
                                                  use_birnn_method2=args.use_birnn_method2)
                    else:
                        if args.use_regularizer:
                            x_rec, x_pred, z_regularizer = model(x, use_mem=args.use_mem, use_pred=args.use_pred,
                                                                 use_reg=args.use_regularizer,
                                                                 use_birnn=args.use_birnn,
                                                                 use_birnn_method2=args.use_birnn_method2)
                        else:
                            x_rec, x_pred = model(x, use_mem=args.use_mem, use_pred=args.use_pred,
                                                  use_reg=args.use_regularizer,
                                                  use_birnn=args.use_birnn,
                                                  use_birnn_method2=args.use_birnn_method2)
                else:
                    x_rec, x_pred = model(x, use_mem=args.use_mem, use_pred=args.use_pred, use_birnn=args.use_birnn)

                if args.use_birnn:
                    x_pred_forward = x_pred[0].permute(0, 2, 1)
                    x_pred_backward = x_pred[1].permute(0, 2, 1)
                    data_forward_pred_loss = pred_criterion(x_pred_forward, pred_target_forward)
                    data_backward_pred_loss = pred_criterion(x_pred_backward, pred_target_backward)
                    data_pred_loss = tuple([data_forward_pred_loss, data_backward_pred_loss])
                else:
                    x_pred = x_pred.permute(0, 2, 1)
                    data_pred_loss = pred_criterion(x_pred, pred_target)

                data_rec_loss = nn.MSELoss(reduction='none')(x_rec, x)

                if args.use_birnn:
                    if args.use_mem:
                        if args.use_regularizer:
                            score = data_rec_loss[:, :, -1].mean(dim=1) + args.forwardpred_weight * data_pred_loss[
                                0].mean(dim=1) + \
                                    args.backwardpred_weight * data_pred_loss[1].mean(
                                dim=1) + args.regularizer_weight * z_regularizer
                        else:
                            score = data_rec_loss[:, :, -1].mean(dim=1) + args.forwardpred_weight * data_pred_loss[
                                0].mean(dim=1) + \
                                    args.backwardpred_weight * data_pred_loss[1].mean(dim=1)
                    else:
                        score = data_rec_loss[:, :, -1].mean(dim=1) + args.forwardpred_weight * data_pred_loss[0].mean(
                            dim=1) + \
                                args.backwardpred_weight * data_pred_loss[1].mean(dim=1)
                else:
                    if args.use_mem:
                        if args.use_regularizer:
                            score = data_rec_loss[:, :, -1].mean(dim=1) + args.pred_weight * data_pred_loss.mean(
                                dim=1) + args.regularizer_weight * z_regularizer
                        else:
                            score = data_rec_loss[:, :, -1].mean(dim=1) + args.pred_weight * data_pred_loss.mean(dim=1)
                    else:
                        score = data_rec_loss[:, :, -1].mean(dim=1) + args.pred_weight * data_pred_loss.mean(dim=1)

            else:
                y = tails[0]
                if args.use_mem:
                    if args.use_regularizer:
                        x_rec, z_regularizer = model(x, use_mem=args.use_mem, use_reg=args.use_mem)
                        data_rec_loss = nn.MSELoss(reduction='none')(x_rec, x)
                        # score = data_rec_loss[:, -1]
                        score = data_rec_loss[:, :, -1].mean(dim=1) + args.regularizer_weight * z_regularizer
                    else:
                        x_rec = model(x, use_mem=args.use_mem)
                        data_rec_loss = nn.MSELoss(reduction='none')(x_rec, x)
                        score = data_rec_loss[:, :, -1].mean(dim=1)
                else:
                    x_rec = model(x, use_mem=args.use_mem)
                    data_rec_loss = nn.MSELoss(reduction='none')(x_rec, x)
                    score = data_rec_loss[:, :, -1].mean(dim=1)

            y_pred.append(score.cpu().numpy().reshape(-1))
            y_true.append(y[:, -1].cpu().numpy().reshape(-1))

    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    if args.data_category == 'SMAP' or args.data_category == 'MSL' or args.data_category == 'Swat':
        performance = calculate_performance(y_true, y_pred, delay=delay, data_category=args.data_category)
    else:
        performance = smd_calculate_performance(y_true, y_pred, data_category=args.data_category)

    global best_performance
    if best_performance is None or performance.loc[0, 'F1'] > best_performance.loc[0, 'F1']:
        best_performance = performance
        torch.save(model.state_dict(), os.path.join(args.save_path, f'model_best.pth.tar'))

    return performance


if __name__ == '__main__':
    args = parse_args()

    if args.seed is not None:
        setup_seed(args.seed)

    # torch.cuda.set_device(args.num_gpu)
    # Preparing
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    best_performance = None

    # Reading dataset
    print(f'[INFO] Reading dataset...')
    # train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m = prepare_dataset(
    #     args.data_path, args.data_category, args.train_val_test_split,
    #     args.label_portion, args.standardization_method, args.filling_method)

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
                                                use_pred=args.use_pred, pred_steps=args.pred_steps,
                                                use_birnn=args.use_birnn,
                                                birnn_method2=args.use_birnn_method2)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                  drop_last=True, pin_memory=True)

        # # Validation dataset
        # val_dataset = KPIBatchedWindowDataset(val_x, label=val_y, mask=val_m, window_size=args.window_size,
        #                                       use_pred=args.use_pred, pred_steps=args.pred_steps)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True,
        #                         pin_memory=True)

        test_dataset = KPIBatchedWindowDataset(x_test, label=y_test, mask=None, window_size=args.window_size,
                                               use_pred=args.use_pred, pred_steps=args.pred_steps,
                                               use_birnn=args.use_birnn,
                                               birnn_method2=args.use_birnn_method2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)
    else:
        train_dataset = KPIBatchedWindowDataset2(x_train, label=None, mask=None, window_size=args.window_size,
                                                 use_pred=args.use_pred, pred_steps=args.pred_steps,
                                                 use_birnn=args.use_birnn,
                                                 birnn_method2=args.use_birnn_method2)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                  drop_last=True, pin_memory=True)

        # # Validation dataset
        # val_dataset = KPIBatchedWindowDataset(val_x, label=val_y, mask=val_m, window_size=args.window_size,
        #                                       use_pred=args.use_pred, pred_steps=args.pred_steps)
        # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True,
        #                         pin_memory=True)

        test_dataset = KPIBatchedWindowDataset2(x_test, label=y_test, mask=None, window_size=args.window_size,
                                                use_pred=args.use_pred, pred_steps=args.pred_steps,
                                                use_birnn=args.use_birnn,
                                                birnn_method2=args.use_birnn_method2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                 drop_last=False,
                                 pin_memory=True)

    # Models
    # if args.variant == 'conv':
    #     encoder = ConvEncoder(args.window_size, 1, args.latent_size).cuda()
    #     decoder = ConvDecoder(args.window_size, 1, args.latent_size).cuda()
    # elif args.variant == 'dense':
    #     encoder = DenseEncoder(args.window_size, args.hidden_size, args.latent_size).cuda()
    #     decoder = DenseDecoder(args.window_size, args.hidden_size, args.latent_size).cuda()
    # else:
    #     raise ValueError('Invalid model variant!')
    # data_discriminator = DataDiscriminator(args.window_size, args.hidden_size).cuda()
    # latent_discriminator = LatentDiscriminator(args.hidden_size, args.latent_size).cuda()

    # Define trainer
    # trainer = RPAD(encoder, decoder, data_discriminator, latent_discriminator, args.batch_size,
    #                args.window_size)
    model = RPAD(window_size=args.window_size, hidden_size=args.hidden_size, latent_size=args.hidden_size,
                 batch_size=args.batch_size, variant=args.variant, use_mem=args.use_mem, use_reg=args.use_regularizer,
                 mem_size=args.mem_size,
                 use_pred=args.use_pred, pred_steps=args.pred_steps, in_channel=x_train.shape[1],
                 use_birnn=args.use_birnn,
                 use_birnn_method2=args.use_birnn_method2)
    #罗注释：在cude环境需要
    # model = model.cuda()
    if args.print_model:
        print(model)

    # Load models
    if args.resume:
        assert args.load_path is not None
        model.load_state_dict(torch.load(args.load_path))
        # check_point = torch.load(args.save_path)
        # print_blue_info('Resume at epoch %d...' % check_point['epoch'])
        # encoder.load_state_dict(check_point['encoder_state_dict'])
        # decoder.load_state_dict(check_point['decoder_state_dict'])
        # data_discriminator.load_state_dict(check_point['data_discriminator_state_dict'])
        # latent_discriminator.load_state_dict(check_point['latent_discriminator_state_dict'])

    # Don't use delay, new USAD evaluate metric
    if args.delay is None:
        delay = 7 if x_train.shape[0] > 80000 else 3
    else:
        delay = args.delay

    train(model, train_loader, test_loader, args)
    torch.save(model.state_dict(), os.path.join(args.save_path, f'model_final.pth.tar'))

    # Evaluating
    # to be test
    print_blue_info('Start evaluating...')
    performance = evaluate(model, test_loader, delay, args)
    print_blue_info('Best performance...')
    save_path = args.save_path
    args = vars(args)
    if args['data_category'] == 'SMAP' or args['data_category'] == 'MSL' or args['data_category'] == 'Swat':
        args['PR_ORI'] = best_performance['PR_ORI'][0]
        args['REC_ORI'] = best_performance['REC_ORI'][0]
        args['F1_ORI'] = best_performance['F1_ORI'][0]
        args['PR'] = best_performance['PR'][0]
        args['REC'] = best_performance['REC'][0]
        args['F1'] = best_performance['F1'][0]
        args['ROC_ORI'] = best_performance['ROC_ORI'][0]
        args['mAP_ORI'] = best_performance['mAP_ORI'][0]
        args['ROC'] = best_performance['ROC'][0]
        args['mAP'] = best_performance['mAP'][0]
        args['train_val_test_split'] = str(args['train_val_test_split'])
    else:
        args['F1_ORI'] = best_performance['F1_ORI'][0]
        args['TP_ORI'] = best_performance['TP_ORI'][0]
        args['FP_ORI'] = best_performance['FP_ORI'][0]
        args['FN_ORI'] = best_performance['FN_ORI'][0]

        args['F1'] = best_performance['F1'][0]
        args['TP'] = best_performance['TP'][0]
        args['FP'] = best_performance['FP'][0]
        args['FN'] = best_performance['FN'][0]
        args['train_val_test_split'] = str(args['train_val_test_split'])

    df = pd.DataFrame(args, index=[0])
    if args['data_category'] == 'SMAP':
        df.to_csv(os.path.join(save_path, f'SMAP_statistics.csv'), index=False)
    elif args['data_category'] == 'MSL':
        df.to_csv(os.path.join(save_path, f'MSL_statistics.csv'), index=False)
    elif args['data_category'] == 'Swat':
        df.to_csv(os.path.join(save_path, f'Swat_statistics.csv'), index=False)
    else:
        df.to_csv(os.path.join(save_path, f'statistics.csv'), index=False)
