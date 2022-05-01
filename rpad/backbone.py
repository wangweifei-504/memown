import numpy as np

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, data_size, input_channel, latent_size):
        super(ConvEncoder, self).__init__()
        self.__dict__.update(locals())

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []

        for i in range(self.layer_num + 1):
            current_out_channel = self.max_channel_num // 2 ** (self.layer_num - i)
            if i == 0:
                self.conv_list.append(nn.Conv1d(in_channels=self.input_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
            else:
                self.conv_list.append(nn.Conv1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
                self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channel = current_out_channel

        self.conv_layers = nn.Sequential(*self.conv_list)

        self.linear_layers = nn.Sequential(
            nn.Linear(self.final_size * self.max_channel_num, latent_size)
        )

    def forward(self, x):

        # out = torch.unsqueeze(x, dim=1)
        # need test
        # out = x.permute(0, 2, 1)
        out = self.conv_layers(x)
        # print("enconder_out_shape:")
        # print(out.shape)
        out = out.view(-1, self.final_size * self.max_channel_num)
        out = self.linear_layers(out)

        return out


class ConvDecoder(nn.Module):
    def __init__(self, data_size, input_channel, latent_size):
        super(ConvDecoder, self).__init__()
        self.__dict__.update(locals())

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []

        self.linear_layers = nn.Sequential(
            nn.Linear(latent_size, self.final_size * self.max_channel_num),
            nn.ReLU(True)
        )

        prev_channel = self.max_channel_num
        for i in range(self.layer_num):
            current_out_channel = self.max_channel_num // 2 ** (i + 1)
            self.conv_list.append(nn.ConvTranspose1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                     kernel_size=4, stride=2, padding=1))
            self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.ReLU(True))
            prev_channel = current_out_channel

        self.conv_list.append(nn.ConvTranspose1d(in_channels=current_out_channel, out_channels=input_channel,
                                                 kernel_size=4, stride=2, padding=1))
        self.conv_list.append(nn.Tanh())

        self.conv_layers = nn.Sequential(*self.conv_list)

    def forward(self, z):
        # print("z.shape:")
        # print(z.shape)
        out = self.linear_layers(z)
        out = out.view(-1, self.max_channel_num, self.final_size)
        out = self.conv_layers(out)
        # print("deconder_out_shape:")
        # print(out.shape)

        # out = torch.squeeze(out, dim=1)

        return out


class ConvPredictDecoder(nn.Module):
    def __init__(self, data_size, input_channel, latent_size, hidden_size, pred_steps):
        super(ConvPredictDecoder, self).__init__()
        self.__dict__.update(locals())

        self.window_size = data_size
        self.input_channel = input_channel
        self.predict_steps = pred_steps

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []

        self.linear_layers = nn.Sequential(
            nn.Linear(latent_size, self.final_size * self.max_channel_num),
            nn.ReLU(True)
        )

        prev_channel = self.max_channel_num
        for i in range(self.layer_num):
            current_out_channel = self.max_channel_num // 2 ** (i + 1)
            self.conv_list.append(nn.ConvTranspose1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                     kernel_size=4, stride=2, padding=1))
            self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.ReLU(True))
            prev_channel = current_out_channel

        self.conv_list.append(nn.ConvTranspose1d(in_channels=current_out_channel, out_channels=input_channel,
                                                 kernel_size=4, stride=2, padding=1))
        self.conv_list.append(nn.Tanh())
        # output the dataset channels
        self.conv_layers = nn.Sequential(*self.conv_list)
        # now output the predict_step length series(windows->predict step)
        self.predict_layers = nn.Sequential(
            nn.Linear(self.window_size * self.input_channel, self.predict_steps * self.input_channel)
        )

    def forward(self, z):
        # print("z.shape:")
        # print(z.shape)
        out = self.linear_layers(z)
        out = out.view(-1, self.max_channel_num, self.final_size)
        out = self.conv_layers(out)

        out = out.view(-1, self.window_size * self.input_channel)
        out = self.predict_layers(out)
        out = out.view(-1, self.predict_steps, self.input_channel)

        # print("predict_out_shape:")
        # print(out.shape)

        # out = torch.squeeze(out, dim=1)

        return out


class RnnPredictor(nn.Module):
    def __init__(self, data_size, input_channel, latent_size, hidden_size, pred_steps):
        super(RnnPredictor, self).__init__()
        self.__dict__.update(locals())

        self.window_size = data_size
        self.hidden_size = hidden_size
        self.input_channel = input_channel
        self.predict_steps = pred_steps

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []
        self.num_layers = 2

        self.linear_layers = nn.Sequential(
            nn.Linear(latent_size, self.final_size * self.max_channel_num),
            nn.ReLU(True)
        )

        prev_channel = self.max_channel_num
        for i in range(self.layer_num):
            current_out_channel = self.max_channel_num // 2 ** (i + 1)
            self.conv_list.append(nn.ConvTranspose1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                     kernel_size=4, stride=2, padding=1))
            self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.ReLU(True))
            prev_channel = current_out_channel

        self.conv_list.append(nn.ConvTranspose1d(in_channels=current_out_channel, out_channels=input_channel,
                                                 kernel_size=4, stride=2, padding=1))
        self.conv_list.append(nn.Tanh())
        # output the dataset channels
        self.conv_layers = nn.Sequential(*self.conv_list)
        # now output the predict_step length series(windows->predict step)
        # input(seq_len, batch, input_size)
        self.predict_layers = nn.LSTM(input_channel, hidden_size, self.num_layers, dropout=0.4)

        self.pred_linear = nn.Sequential(
            nn.Linear(self.window_size * self.hidden_size, self.predict_steps * self.input_channel)
        )

    def forward(self, z):
        # print("z.shape:")
        # print(z.shape)
        out = self.linear_layers(z)
        out = out.view(-1, self.max_channel_num, self.final_size)
        out = self.conv_layers(out)
        # print("out::::::::::::::::::::::::::::::")
        # print(out)
        # print("out.shape::::::::::::::::::::::::::::::::::::")
        # print(out.shape)
        # print("out.shape::::::::::::::::::::::::::::::::::::")
        out = out.permute(2, 0, 1)
        # print("input_lstm_shape::::::::::::::::::::::")
        # print(out.shape)

        # out = out.view(-1, self.window_size * self.input_channel)
        # print("z.shape:::::::::::::::::::::::::::::::::::")
        # print(z.shape)

        batch_size = z.shape[0]

        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).cuda(non_blocking=True)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).cuda(non_blocking=True)
        # h0 = torch.randn(self.num_layers, batch_size,
        #                  self.hidden_size).cuda(non_blocking=True)
        # c0 = torch.randn(self.num_layers, batch_size,
        #                  self.hidden_size).cuda(non_blocking=True)
        out, _ = self.predict_layers(out, (h0, c0))
        # print("out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)
        out = out.view(-1, self.window_size * self.hidden_size)
        # print("out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)

        out = self.pred_linear(out)
        # print("out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)
        out = out.view(-1, self.predict_steps, self.input_channel)
        # print("out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)

        # print("predict_out_shape:")
        # print(out.shape)

        # out = torch.squeeze(out, dim=1)

        return out


class Bi_RnnPredictor(nn.Module):

    def __init__(self, data_size, input_channel, latent_size, hidden_size, pred_steps):
        super(Bi_RnnPredictor, self).__init__()
        self.__dict__.update(locals())

        self.window_size = data_size
        self.hidden_size = hidden_size
        self.input_channel = input_channel
        self.predict_steps = pred_steps

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []
        self.num_layers = 2
        self.num_directions = 2

        self.linear_layers = nn.Sequential(
            nn.Linear(latent_size, self.final_size * self.max_channel_num),
            nn.ReLU(True)
        )

        prev_channel = self.max_channel_num
        for i in range(self.layer_num):
            current_out_channel = self.max_channel_num // 2 ** (i + 1)
            self.conv_list.append(nn.ConvTranspose1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                     kernel_size=4, stride=2, padding=1))
            self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.ReLU(True))
            prev_channel = current_out_channel

        self.conv_list.append(nn.ConvTranspose1d(in_channels=current_out_channel, out_channels=input_channel,
                                                 kernel_size=4, stride=2, padding=1))
        self.conv_list.append(nn.Tanh())
        # output the dataset channels
        self.conv_layers = nn.Sequential(*self.conv_list)
        # now output the predict_step length series(windows->predict step)
        # input(seq_len, batch, input_size)
        self.predict_layers = nn.LSTM(input_channel, hidden_size, self.num_layers, dropout=0.4, bidirectional=True)

        self.pred_linear = nn.Sequential(
            nn.Linear(self.window_size * self.hidden_size, self.predict_steps * self.input_channel)
        )

    def forward(self, z):
        # print("z.shape:")
        # print(z.shape)
        out = self.linear_layers(z)
        out = out.view(-1, self.max_channel_num, self.final_size)
        out = self.conv_layers(out)
        # print("out::::::::::::::::::::::::::::::")
        # print(out)
        # print("out.shape::::::::::::::::::::::::::::::::::::")
        # print(out.shape)
        # print("out.shape::::::::::::::::::::::::::::::::::::")
        out = out.permute(2, 0, 1)
        # print(out.shape)

        # out = out.view(-1, self.window_size * self.input_channel)
        # print("z.shape:::::::::::::::::::::::::::::::::::")
        # print(z.shape)

        batch_size = z.shape[0]

        h0 = torch.zeros(self.num_layers * 2, batch_size,
                         self.hidden_size).cuda(non_blocking=True)
        c0 = torch.zeros(self.num_layers * 2, batch_size,
                         self.hidden_size).cuda(non_blocking=True)
        # h0 = torch.randn(self.num_layers, batch_size,
        #                  self.hidden_size).cuda(non_blocking=True)
        # c0 = torch.randn(self.num_layers, batch_size,
        #                  self.hidden_size).cuda(non_blocking=True)
        # print("out before:::::::::::::::::::::::::::")
        # print(out.shape)

        out, _ = self.predict_layers(out, (h0, c0))
        # print("after lstm::::::::::::::::::::::::::::")
        # print(out.shape)

        # print("lstm_out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)
        forward_and_back_seq = out.view(self.window_size, batch_size, self.num_directions, self.hidden_size)
        forward_seq = forward_and_back_seq[:, :, 0, :]
        backward_seq = forward_and_back_seq[:, :, 1, :]
        # print("forward_and_back_seq.shape::::::::::::::::::::::::::::::::::::")
        # print(forward_and_back_seq.shape)

        # print("forward_shape:::::::::::::::::::::::::::::::::")
        # print(forward_seq.shape)
        # print("backward_shape:::::::::::::::::::::::::::::::::")
        # print(backward_seq.shape)

        forward_seq = forward_seq.contiguous().view(-1, self.window_size * self.hidden_size)
        backward_seq = backward_seq.contiguous().view(-1, self.window_size * self.hidden_size)
        # out = out.view(-1, self.window_size * self.hidden_size)
        forward_seq = self.pred_linear(forward_seq)
        backward_seq = self.pred_linear(backward_seq)

        forward_seq = forward_seq.view(-1, self.predict_steps, self.input_channel)
        backward_seq = backward_seq.view(-1, self.predict_steps, self.input_channel)

        # print("return forward_seq::::::::::::::::::::::")
        # print(forward_seq.shape)
        return forward_seq, backward_seq

        # out = self.pred_linear(out)
        # print("out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)
        # out = out.view(-1, self.predict_steps, self.input_channel)
        # print("out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)

        # print("predict_out_shape:")
        # print(out.shape)

        # out = torch.squeeze(out, dim=1)


class Bi_RnnPredictor2(nn.Module):

    def __init__(self, data_size, input_channel, latent_size, hidden_size, pred_steps):
        super(Bi_RnnPredictor2, self).__init__()
        self.__dict__.update(locals())

        self.window_size = data_size
        self.hidden_size = hidden_size
        self.input_channel = input_channel
        self.predict_steps = pred_steps

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []
        self.num_layers = 2
        self.num_directions = 1

        self.linear_layers = nn.Sequential(
            nn.Linear(latent_size, self.final_size * self.max_channel_num),
            nn.ReLU(True)
        )

        prev_channel = self.max_channel_num
        for i in range(self.layer_num):
            current_out_channel = self.max_channel_num // 2 ** (i + 1)
            self.conv_list.append(nn.ConvTranspose1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                     kernel_size=4, stride=2, padding=1))
            self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.ReLU(True))
            prev_channel = current_out_channel

        self.conv_list.append(nn.ConvTranspose1d(in_channels=current_out_channel, out_channels=input_channel,
                                                 kernel_size=4, stride=2, padding=1))
        self.conv_list.append(nn.Tanh())
        # output the dataset channels
        self.conv_layers = nn.Sequential(*self.conv_list)
        # now output the predict_step length series(windows->predict step)
        # input(seq_len, batch, input_size)
        self.predict_layers = nn.LSTM(input_channel, hidden_size, self.num_layers, dropout=0.4)
        # self.predict_layer_forward = nn.LSTM(input_channel, hidden_size, self.num_layers, dropout=0.4)
        # self.predict_layer_backward = nn.LSTM(input_channel, hidden_size, self.num_layers, dropout=0.4)

        self.pred_linear = nn.Sequential(
            nn.Linear(self.window_size * self.hidden_size, self.predict_steps * self.input_channel)
        )

    def forward(self, z):
        # print("z.shape:")
        # print(z.shape)
        out = self.linear_layers(z)
        out = out.view(-1, self.max_channel_num, self.final_size)
        out = self.conv_layers(out)
        # print("out::::::::::::::::::::::::::::::")
        # print(out)
        # print("out.shape::::::::::::::::::::::::::::::::::::")
        # print(out.shape)
        # print("out.shape::::::::::::::::::::::::::::::::::::")
        out = out.permute(2, 0, 1)
        # print(out.shape)

        # out = out.view(-1, self.window_size * self.input_channel)
        # print("z.shape:::::::::::::::::::::::::::::::::::")
        # print(z.shape)

        batch_size = z.shape[0]

        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).cuda(non_blocking=True)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).cuda(non_blocking=True)
        # h0 = torch.randn(self.num_layers, batch_size,
        #                  self.hidden_size).cuda(non_blocking=True)
        # c0 = torch.randn(self.num_layers, batch_size,
        #                  self.hidden_size).cuda(non_blocking=True)
        out, _ = self.predict_layers(out, (h0, c0))
        # print("lstm_out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)

        # forward_and_back_seq = out.view(self.window_size, batch_size, self.num_directions, self.hidden_size)
        # forward_seq = forward_and_back_seq[:, :, 0, :]
        # backward_seq = forward_and_back_seq[:, :, 1, :]
        # print("forward_and_back_seq.shape::::::::::::::::::::::::::::::::::::")
        # print(forward_and_back_seq.shape)

        # print("forward_shape:::::::::::::::::::::::::::::::::")
        # print(forward_seq.shape)
        # print("backward_shape:::::::::::::::::::::::::::::::::")
        # print(backward_seq.shape)

        # forward_seq = forward_seq.contiguous().view(-1, self.window_size * self.hidden_size)
        # backward_seq = backward_seq.contiguous().view(-1, self.window_size * self.hidden_size)
        out = out.view(-1, self.window_size * self.hidden_size)
        out = self.pred_linear(out)
        # forward_seq = self.pred_linear(forward_seq)
        # backward_seq = self.pred_linear(backward_seq)

        pred_seq = out.view(-1, self.predict_steps, self.input_channel)
        # print("pred_seq shape::::::::::::::::::::::::::::::::::::::::::::::")
        # print(pred_seq.shape)

        # print(;)
        # forward_seq = forward_seq.view(-1, self.predict_steps, self.input_channel)
        # backward_seq = backward_seq.view(-1, self.predict_steps, self.input_channel)

        return pred_seq

        # out = self.pred_linear(out)
        # print("out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)
        # out = out.view(-1, self.predict_steps, self.input_channel)
        # print("out.shape::::::::::::::::::::::::::::::::")
        # print(out.shape)

        # print("predict_out_shape:")
        # print(out.shape)

        # out = torch.squeeze(out, dim=1)


class DenseEncoder(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size):
        super(DenseEncoder, self).__init__()
        self.__dict__.update(locals())

        self.layers = nn.Sequential(
            nn.Linear(data_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, x):
        out = self.layers(x)

        return out


class DenseDecoder(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size):
        super(DenseDecoder, self).__init__()
        self.__dict__.update(locals())

        self.layers = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, data_size)
        )

    def forward(self, z):
        out = self.layers(z)

        return out


class LatentDiscriminator(nn.Module):
    def __init__(self, hidden_size, latent_size):
        super(LatentDiscriminator, self).__init__()
        self.__dict__.update(locals())

        self.layers = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

    def forward(self, z):
        out = self.layers(z)

        return out


class DataDiscriminator(nn.Module):
    def __init__(self, data_size, input_channel, hidden_size):
        super(DataDiscriminator, self).__init__()
        self.__dict__.update(locals())

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []

        # self.layers = nn.Sequential(
        #     nn.Linear(data_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_size, 1),
        #     # nn.BatchNorm1d(1),
        #     # nn.Sigmoid()
        # )
        for i in range(self.layer_num + 1):
            current_out_channel = self.max_channel_num // 2 ** (self.layer_num - i)

            if i == 0:
                self.conv_list.append(nn.Conv1d(in_channels=self.input_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
            else:
                self.conv_list.append(nn.Conv1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
                self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channel = current_out_channel

        self.conv_layers = nn.Sequential(*self.conv_list)

        self.linear_layers = nn.Sequential(
            nn.Linear(self.final_size * self.max_channel_num, 1)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, self.final_size * self.max_channel_num)
        out = self.linear_layers(out)
        return out
