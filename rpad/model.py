import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import (
    ConvEncoder,
    ConvDecoder,
    ConvPredictDecoder,
    RnnPredictor,
    Bi_RnnPredictor,
    Bi_RnnPredictor2,
    DenseEncoder,
    DenseDecoder,
    DataDiscriminator,
    LatentDiscriminator
)


# from .util import entropy


class WeightedPredictionLoss(nn.Module):
    # def __init__(self, pred_steps):
    #     super(WeightedPredictionLoss, self).__init__()
    #
    #     self.pred_steps = pred_steps
    #     self.weights = nn.Parameter(torch.arange(pred_steps + 1, 1, -1, dtype=torch.float))
    #
    # def forward(self, inputs, targets):
    #     pred_dev = F.mse_loss(inputs, targets, reduction='none')
    #     pred_loss = torch.einsum('ij,j->i', [pred_dev, self.weights])
    #     pred_loss = pred_loss / self.pred_steps ** 2
    #
    #     return torch.mean(pred_loss)

    def __init__(self, pred_steps, reduction='mean'):
        super(WeightedPredictionLoss, self).__init__()

        self.pred_steps = pred_steps
        self.reduction = reduction
        assert reduction in ['mean', 'none']
        self.weights = nn.Parameter(torch.arange(pred_steps + 1, 1, -1, dtype=torch.float))

    def forward(self, inputs, targets):
        pred_dev = F.mse_loss(inputs, targets, reduction='none')

        pred_loss = torch.einsum('ijk,k->ij', pred_dev, self.weights)

        pred_loss = pred_loss / self.pred_steps ** 2

        if self.reduction == 'mean':
            return torch.mean(pred_loss)
        else:
            return pred_loss


class MemoryModule(nn.Module):
    def __init__(self, mem_size, hidden_size, latent_size):
        super(MemoryModule, self).__init__()

        self.memory_bank = nn.Parameter(torch.randn(mem_size, latent_size))
        self.pred_net = DenseEncoder(latent_size, hidden_size, mem_size)

    def forward(self, z):
        # z = F.softmax(z, dim=-1)
        weights = self.pred_net(z)
        # print("weight::::::::::::::::::::::")
        # print(weights)
        z = F.softmax(weights, dim=-1)
        # print("mem_regularizer::::::::::::::::::::::::::::")
        # print(regularizer)
        z_mem = torch.einsum('ij,ki->kj', self.memory_bank, z)

        return z_mem


class MemoryModuleRegularizer(nn.Module):
    def __init__(self, mem_size, hidden_size, latent_size):
        super(MemoryModuleRegularizer, self).__init__()

        self.memory_bank = nn.Parameter(torch.randn(mem_size, latent_size))
        self.pred_net = DenseEncoder(latent_size, hidden_size, mem_size)

        # self.predictor = ConvDecoder(window_size, in_channel, latent_size)

    def entropy(self, w: torch.Tensor, dim=-1):
        return -(w * torch.log(w)).sum(dim=dim)

    def forward(self, z):
        # z = F.softmax(z, dim=-1)
        weights = self.pred_net(z)
        # print("weight::::::::::::::::::::::")
        # print(weights)
        z = F.softmax(weights, dim=-1)
        regularizer = self.entropy(z + 0.0001)
        # print("mem_regularizer::::::::::::::::::::::::::::")
        # print(regularizer)
        z_mem = torch.einsum('ij,ki->kj', self.memory_bank, z)

        return z_mem, regularizer.mean()


class RPAD(nn.Module):
    def __init__(self, window_size, hidden_size, latent_size, batch_size, variant, use_mem=False, use_reg=False,
                 mem_size=None,
                 use_pred=False, pred_steps=5, in_channel=1, use_birnn=False, use_birnn_method2=False, ):
        super(RPAD, self).__init__()

        if use_mem:
            assert mem_size is not None
        self.use_mem = use_mem
        self.use_pred = use_pred
        self.use_birnn = use_birnn
        self.use_birnn_method2 = use_birnn_method2
        self.use_reg = use_reg

        if variant == 'conv':
            self.encoder = ConvEncoder(window_size, in_channel, latent_size)
            self.decoder = ConvDecoder(window_size, in_channel, latent_size)
        elif variant == 'dense':
            self.encoder = DenseEncoder(window_size, hidden_size, latent_size)
            self.decoder = DenseDecoder(window_size, hidden_size, latent_size)
        else:
            raise ValueError('Invalid model variant!')
        self.data_discriminator = DataDiscriminator(window_size, in_channel, hidden_size)
        self.latent_discriminator = LatentDiscriminator(hidden_size, latent_size)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.reconstruction_criterion = nn.L1Loss(reduction='none')
        self.prediction_criterion = WeightedPredictionLoss(pred_steps)

        self.input_noise = lambda: (torch.randn(batch_size, in_channel, window_size) * 0.01)
        if torch.cuda.is_available():
            self.input_noise = lambda: (torch.randn(batch_size, in_channel, window_size) * 0.01).cuda()
        self.register_buffer("label_real", torch.ones(batch_size, 1))
        self.register_buffer("label_fake", torch.zeros(batch_size, 1))

        if use_mem:
            if use_reg:
                self.memory = MemoryModuleRegularizer(mem_size, hidden_size, latent_size)
            else:
                self.memory = MemoryModule(mem_size, hidden_size, latent_size)

        if use_pred:
            # self.predictor = DenseEncoder(latent_size, hidden_size, pred_steps)
            # output batch_size * pred_steps * multi_dim
            # watch the pred_loss
            # self.predictor = ConvPredictDecoder(window_size, in_channel, latent_size, hidden_size, pred_steps)
            if use_birnn:
                if use_birnn_method2:
                    self.predictor = Bi_RnnPredictor2(window_size, in_channel, latent_size, hidden_size, pred_steps)
                else:
                    self.predictor = Bi_RnnPredictor(window_size, in_channel, latent_size, hidden_size, pred_steps)
            else:
                self.predictor = RnnPredictor(window_size, in_channel, latent_size, hidden_size, pred_steps)
                # self.predictor = ConvPredictDecoder(window_size, in_channel, latent_size, hidden_size, pred_steps)

    # def print_model(self):
    #     print(self.encoder)
    #     print(self.decoder)
    #     print(self.data_discriminator)
    #     print(self.latent_discriminator)

    # def train(self):
    #     self.encoder.train()
    #     self.decoder.train()
    #     self.data_discriminator.train()
    #     self.latent_discriminator.train()

    # def eval(self):
    #     self.encoder.eval()
    #     self.decoder.eval()
    #     self.data_discriminator.eval()
    #     self.latent_discriminator.eval()

    def data_gen_loss(self, x, x_next=None, y=None, y_pred=None, margin=1.0):
        if self.use_pred:
            if self.use_birnn:
                if self.use_birnn_method2:
                    if self.use_reg:
                        x_rec, x_pred, z_regularizer = self.forward(x + self.input_noise(),
                                                                    x_next=x_next + self.input_noise(),
                                                                    use_mem=self.use_mem,
                                                                    use_reg=self.use_reg,
                                                                    use_pred=True,
                                                                    use_birnn=self.use_birnn,
                                                                    use_birnn_method2=self.use_birnn_method2)
                        x_pred_forward = x_pred[0].permute(0, 2, 1)
                        x_pred_backward = x_pred[1].permute(0, 2, 1)
                        pred_forward_loss = self.prediction_criterion(x_pred_forward, y_pred[0])
                        pred_backward_loss = self.prediction_criterion(x_pred_backward, y_pred[1])
                        pred_loss = tuple([pred_forward_loss, pred_backward_loss])
                    else:
                        x_rec, x_pred = self.forward(x + self.input_noise(),
                                                     x_next=x_next + self.input_noise(),
                                                     use_mem=self.use_mem,
                                                     use_reg=self.use_reg,
                                                     use_pred=True,
                                                     use_birnn=self.use_birnn,
                                                     use_birnn_method2=self.use_birnn_method2)
                        x_pred_forward = x_pred[0].permute(0, 2, 1)
                        x_pred_backward = x_pred[1].permute(0, 2, 1)
                        pred_forward_loss = self.prediction_criterion(x_pred_forward, y_pred[0])
                        pred_backward_loss = self.prediction_criterion(x_pred_backward, y_pred[1])
                        pred_loss = tuple([pred_forward_loss, pred_backward_loss])

                else:
                    if self.use_reg:
                        x_rec, x_pred, z_regularizer = self.forward(x + self.input_noise(),
                                                                    use_mem=self.use_mem,
                                                                    use_reg=self.use_reg,
                                                                    use_pred=True,
                                                                    use_birnn=self.use_birnn,
                                                                    use_birnn_method2=self.use_birnn_method2)
                        x_pred_forward = x_pred[0].permute(0, 2, 1)
                        x_pred_backward = x_pred[1].permute(0, 2, 1)
                        pred_forward_loss = self.prediction_criterion(x_pred_forward, y_pred[0])
                        pred_backward_loss = self.prediction_criterion(x_pred_backward, y_pred[1])
                        pred_loss = tuple([pred_forward_loss, pred_backward_loss])
                    else:
                        x_rec, x_pred = self.forward(x + self.input_noise(),
                                                     use_mem=self.use_mem,
                                                     use_reg=self.use_reg,
                                                     use_pred=True,
                                                     use_birnn=self.use_birnn,
                                                     use_birnn_method2=self.use_birnn_method2)
                        x_pred_forward = x_pred[0].permute(0, 2, 1)
                        x_pred_backward = x_pred[1].permute(0, 2, 1)
                        pred_forward_loss = self.prediction_criterion(x_pred_forward, y_pred[0])
                        pred_backward_loss = self.prediction_criterion(x_pred_backward, y_pred[1])
                        pred_loss = tuple([pred_forward_loss, pred_backward_loss])
            else:
                if self.use_reg:
                    x_rec, x_pred, z_regularizer = self.forward(x + self.input_noise(), use_mem=self.use_mem,
                                                                use_pred=True, use_birnn=self.use_birnn)
                else:
                    x_rec, x_pred = self.forward(x + self.input_noise(), use_mem=self.use_mem, use_pred=True,
                                                 use_birnn=self.use_birnn)

                x_pred = x_pred.permute(0, 2, 1)
                # print("x_pred_shape:")
                # print(x_pred.shape)
                # print("y_pred_shape:")
                # print(y_pred.shape)
                pred_loss = self.prediction_criterion(x_pred, y_pred)

            data_gen_loss = self.adversarial_criterion(self.data_discriminator(x_rec), self.label_real)

            if y is None:
                rec_loss = self.adversarial_criterion(x_rec, x).mean()
            else:
                rec = self.reconstruction_criterion(x_rec, x)
                rec_loss = ((1 - y) * rec + y * torch.max(torch.zeros_like(rec), margin - rec)).mean()
            # print("x_pred_shape:")
            # print(x_pred.shape)
            # print("y_pred_shape:")
            # print(y_pred.shape)
            if self.use_reg:
                return data_gen_loss, rec_loss, pred_loss, z_regularizer
            else:
                return data_gen_loss, rec_loss, pred_loss
        else:
            if self.use_reg:
                x_rec, z_regularizer = self.forward(x + self.input_noise(), use_mem=self.use_mem, use_pred=False)
            else:
                x_rec = self.forward(x + self.input_noise(), use_mem=self.use_mem, use_pred=False)

            data_gen_loss = self.adversarial_criterion(self.data_discriminator(x_rec), self.label_real)

            if y is None:
                rec_loss = self.adversarial_criterion(x_rec, x).mean()
            else:
                rec = self.reconstruction_criterion(x_rec, x)
                rec_loss = ((1 - y) * rec + y * torch.max(torch.zeros_like(rec), margin - rec)).mean()
            if self.use_reg:
                return data_gen_loss, rec_loss, z_regularizer
            else:
                return data_gen_loss, rec_loss

    def data_dis_loss(self, x, x_next=None):
        if self.use_birnn_method2:
            x_rec, *_ = self.forward(x + self.input_noise(), x_next=x_next, use_mem=self.use_mem, use_reg=self.use_reg,
                                     use_pred=self.use_pred, use_birnn=self.use_birnn,
                                     use_birnn_method2=self.use_birnn_method2)
        else:
            if self.use_pred:
                x_rec, *_ = self.forward(x + self.input_noise(), use_mem=self.use_mem, use_reg=self.use_reg,
                                         use_pred=self.use_pred, use_birnn=self.use_birnn,
                                         use_birnn_method2=self.use_birnn_method2)
            else:
                x_rec = self.forward(x + self.input_noise(), use_mem=self.use_mem, use_reg=self.use_reg,
                                     use_pred=self.use_pred,
                                     use_birnn=self.use_birnn, use_birnn_method2=self.use_birnn_method2)

        data_dis_loss = self.adversarial_criterion(self.data_discriminator(x_rec), self.label_fake) + \
                        self.adversarial_criterion(self.data_discriminator(x), self.label_real)

        return data_dis_loss

    def latent_gen_loss(self, x):
        z_hat = self.encoder(x + self.input_noise())
        latent_gen_loss = self.adversarial_criterion(self.latent_discriminator(z_hat), self.label_real)

        return latent_gen_loss

    def latent_dis_loss(self, x):
        z_hat = self.encoder(x + self.input_noise())
        #在cude环境需要
        #z_prior = torch.randn_like(z_hat).cuda()
        z_prior = torch.randn_like(z_hat)
        latent_dis_loss = self.adversarial_criterion(self.latent_discriminator(z_hat),
                                                     self.label_fake) + self.adversarial_criterion(
            self.latent_discriminator(z_prior), self.label_real)

        return latent_dis_loss

    # def predict(self, x):

    # def reconstruct(self, x):
    #     out = self.decoder(self.encoder(x))
    #
    #     return out

    # def save(self, path, name):
    #     print_blue_info('Saving checkpoints...')
    #     torch.save({'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict(),
    #                 'data_discriminator': self.data_discriminator.state_dict(),
    #                 'latent_discriminator': self.latent_discriminator.state_dict()
    #                 }, os.path.join(path, name))
    #
    # def load(self, path):
    #     print_blue_info('Reading from checkpoints...')
    #     data_dict = torch.load(path)
    #     self.encoder.load_state_dict(data_dict['encoder'])
    #     self.decoder.load_state_dict(data_dict['decoder'])
    #     self.data_discriminator.load_state_dict(data_dict['data_discriminator'])
    #     self.latent_discriminator.load_state_dict(data_dict['latent_discriminator'])

    def forward(self, x, x_next=None, use_mem=False, use_reg=False, use_pred=False, use_birnn=False,
                use_birnn_method2=False):
        if use_mem:
            if use_reg:
                z = self.encoder(x)
                z_mem, z_regularizer = self.memory(z)
                x_rec = self.decoder(z_mem)
                if use_pred:
                    if use_birnn:
                        if use_birnn_method2:
                            with torch.no_grad():
                                z_nextwinddow = self.encoder(x_next)
                                z_nextmem, _ = self.memory(z_nextwinddow)
                            forward_pred = self.predictor(z_mem)
                            backward_pred = self.predictor(z_nextmem)
                            pred = tuple([forward_pred, backward_pred])
                            return x_rec, pred, z_regularizer
                        else:
                            forward_pred, backward_pred = self.predictor(z_mem)
                            pred = tuple([forward_pred, backward_pred])
                            return x_rec, pred, z_regularizer
                    else:
                        x_pred = self.predictor(z_mem)
                        return x_rec, x_pred, z_regularizer
                else:
                    return x_rec, z_regularizer
            else:
                z = self.encoder(x)
                z_mem = self.memory(z)
                x_rec = self.decoder(z_mem)
                if use_pred:
                    if use_birnn:
                        if use_birnn_method2:
                            with torch.no_grad():
                                z_nextwinddow = self.encoder(x_next)
                                z_nextmem = self.memory(z_nextwinddow)
                            forward_pred = self.predictor(z_mem)
                            backward_pred = self.predictor(z_nextmem)
                            pred = tuple([forward_pred, backward_pred])
                            return x_rec, pred
                        else:
                            forward_pred, backward_pred = self.predictor(z_mem)
                            pred = tuple([forward_pred, backward_pred])
                            return x_rec, pred
                    else:
                        x_pred = self.predictor(z_mem)
                        return x_rec, x_pred
                else:
                    return x_rec

        else:
            z = self.encoder(x)
            x_rec = self.decoder(z)
            if use_pred:
                if use_birnn:
                    if use_birnn_method2:
                        with torch.no_grad():
                            z_nextwinddow = self.encoder(x_next)
                        forward_pred = self.predictor(z)
                        backward_pred = self.predictor(z_nextwinddow)
                        pred = tuple([forward_pred, backward_pred])
                        return x_rec, pred
                    else:
                        forward_pred, backward_pred = self.predictor(z)
                        pred = tuple([forward_pred, backward_pred])
                        return x_rec, pred
                else:
                    x_pred = self.predictor(z)
                    return x_rec, x_pred
            else:
                return x_rec

        return out
