import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sbibm.utils.nflows import get_flow


class Cond_MADE(nn.Module):
    def __init__(self, dim_in, dim_out, n_hidden, device, random_order=False, random_degree=False, residual=False):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        :param random_order: Whether to use random input order, default is False.
        :param random_degree: Whether to use random degree, default is False.
        :param residual: Whether to enable residual structure.
        """

        super(Cond_MADE, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out  # for gaussian mu and log-sigma output
        self.dim_condition = dim_in - dim_out
        self.n_hidden = n_hidden
        # self.act_func = nn.ReLU()
        self.act_func = nn.Tanh()
        self.random_order = random_order
        self.random_degree = random_degree
        self.device = device
        # self.residual = residual

        self.degrees = self.create_degrees(dim_out, n_hidden, random_order, random_degree)  # only connect x to first hidden layer!!
        # add degrees for conditional param
        self.degrees[0] = np.concatenate(([0] * (dim_in - dim_out), self.degrees[0])).astype('int32')
        # self.mask_matrix = self.create_mask(self.degrees)

        dim_list = [self.dim_in, *n_hidden, self.dim_out * 2]
        self.layers = []
        for i in range(len(dim_list) - 2):
            self.layers.append(MaskedLinear(dim_list[i], dim_list[i + 1]), )
            self.layers.append(self.act_func)
        self.layers.append(MaskedLinear(dim_list[-2], dim_list[-1]))
        self.model = nn.Sequential(*self.layers)
        mask_matrix = self.create_mask(self.degrees)
        mask_iter = iter(mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(torch.tensor(next(mask_iter).transpose(), device=self.device))

    def create_degrees(self, dim_in, n_hidden, random_order, random_degree):
        # for p(theta|x), only connect x to first hidden layer
        degrees = []
        # create degrees for inputs
        if isinstance(random_order, bool):
            if random_order:
                degrees_0 = np.arange(1, dim_in + 1)
                np.random.shuffle(degrees_0[self.dim_condition:])
            else:
                degrees_0 = np.arange(1, dim_in + 1)

        else:
            input_order = np.array(random_order)
            assert np.all(np.sort(input_order) == np.arange(1, dim_in + 1)), 'invalid input order'
            degrees_0 = input_order
        degrees.append(degrees_0)
        # create degrees for hiddens
        if random_degree:
            for N in n_hidden:
                min_prev_degree = min(np.min(degrees[-1]), dim_in - 1)
                degrees_l = np.random.randint(min_prev_degree, dim_in, N)
                degrees.append(degrees_l)
        else:
            for N in n_hidden:
                degrees_l = np.arange(N) % max(1, dim_in - 1) + min(1, dim_in - 1)
                degrees.append(degrees_l)
        if random_degree:
            pass
        return degrees

    def create_mask(self, degrees):
        Ms = []
        for l, (d0, d1) in enumerate(zip(degrees[:-1], degrees[1:])):
            Ms.append(d0[:, np.newaxis] <= d1)
        last_mat = (degrees[-1][:, np.newaxis] < degrees[0])[:, self.dim_condition:]
        Ms.append(np.concatenate((last_mat, last_mat), axis=1))
        return Ms

    def set_masked_linear(self):
        mask_iter = iter(self.mask_matrix)
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module.initialise_mask(torch.tensor(next(mask_iter).transpose(), device=self.device))

    def forward(self, x):
        return self.model(x)


class MaskedLinear(nn.Linear):
    def __init__(self, n_in: int, n_out: int, bias: bool = True) -> None:
        super().__init__(n_in, n_out, bias)
        self.mask = None

    def initialise_mask(self, mask):
        # mask shape: (out_features, in_features)
        self.mask = mask

    def forward(self, x):
        # overrride return F.linear(input, self.weight, self.bias)
        return F.linear(x, self.mask * self.weight, self.bias)


class BatchNormLayer(nn.Module):
    def __init__(self, dim_in, dim_out, eps=1e-5):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim_out))
        self.beta = nn.Parameter(torch.zeros(1, dim_out))
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x):
        x_part_1 = x[:, :(self.dim_in - self.dim_out)]
        x_part_2 = x[:, (self.dim_in - self.dim_out):]
        x_hat, log_det = self._forward(x_part_2)
        return torch.cat((x_part_1, x_hat), dim=1), log_det

    def _forward(self, x):
        # x[(self.dim_in - self.dim_out):]
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps  # torch.mean((x - m) ** 2, axis=0) + self.eps
            # v = torch.mean((x - m) ** 2, dim=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean.clone()
            v = self.batch_var.clone()

        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v))
        return x_hat, log_det

    def backward(self, x):
        x_part_1 = x[:, :(self.dim_in - self.dim_out)]
        x_part_2 = x[:, (self.dim_in - self.dim_out):]
        x_hat, log_det = self._backward(x_part_2)
        return torch.cat((x_part_1, x_hat), dim=1), log_det

    def _backward(self, x):
        if self.training:
            m = x.mean(dim=0)
            v = x.var(dim=0) + self.eps
            self.batch_mean = None
        else:
            if self.batch_mean is None:
                self.set_batch_stats_func(x)
            m = self.batch_mean
            v = self.batch_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det

    def set_batch_stats_func(self, x):
        # print("setting batch stats for validation")
        self.batch_mean = x.mean(dim=0)
        self.batch_var = x.var(dim=0) + self.eps


class Cond_MAF_Layer(nn.Module):
    def __init__(self, dim_in, dim_out, n_hidden, device, reverse=True, random_order=False, random_degree=False, residual=False):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        :param reverse: Whether to reverse input in each MADE.
        :param random_order: Whether to use random input order, default is False.
        :param random_degree: Whether to use random degree, default is False.
        :param residual: Whether to enable residual structure.
        """

        super(Cond_MAF_Layer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_hidden = n_hidden
        self.device = device
        self.reverse = reverse
        self.random_order = random_order
        self.random_degree = random_degree
        self.residual = residual
        self.made = Cond_MADE(dim_in, dim_out, n_hidden, device, random_order, random_degree, residual)
        self.param_trun = True

    def forward(self, x):
        mu, logs = torch.chunk(self.made(x), 2, dim=1)
        if self.param_trun:
            mu = torch.clamp(mu, min=-100, max=100)
            logs = torch.clamp(logs, min=-20, max=20)
        # u = (x[:, (self.dim_in - self.dim_out):] - mu) * torch.exp(-logs + 1e-7)
        u = (x[:, (self.dim_in - self.dim_out):] - mu) * torch.exp(-logs)
        if self.reverse:
            x = torch.cat((x[:, 0:(self.dim_in - self.dim_out)].flip(dims=(1,)), u.flip(dims=(1,))), dim=1)
        else:
            x = torch.cat((x[:, 0:(self.dim_in - self.dim_out)], u), dim=1)
        return x, - torch.sum(logs, dim=1)

    def backward(self, u):
        if self.reverse:
            u = torch.cat((u[:, 0:(self.dim_in - self.dim_out)].flip(dims=(1,)),
                           u[:, (self.dim_in - self.dim_out):].flip(dims=(1,))), dim=1)
        x = torch.zeros_like(u)
        # print('backward fun called')
        x[:, 0:(self.dim_in - self.dim_out)] = u[:, 0:(self.dim_in - self.dim_out)]
        for dim in range(self.dim_out):
            mu, logs = torch.chunk(self.made(x), 2, dim=1)
            if self.param_trun:
                mu = torch.clamp(mu, min=-100, max=100)
                logs = torch.clamp(logs, min=-20, max=20)
            x[:, (dim + self.dim_in - self.dim_out)] = mu[:, dim] + u[:, (dim + self.dim_in - self.dim_out)] * torch.exp(logs[:, dim])
        log_det = torch.sum(logs, dim=1)
        return x, log_det


class Cond_MAF(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_hidden, device, batch_norm=False,
                 reverse=True, random_order=False, random_degree=False, residual=False):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_layer: layer size of MADE
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        :param batch_norm: Whether to enable batch normalization
        :param reverse: Whether to reverse input in each MADE.
        :param random_order: Whether to use random input order, default is False.
        :param random_degree: Whether to use random degree, default is False.
        :param residual: Whether to enable residual structure.
        """

        super(Cond_MAF, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.device = device
        self.reverse = reverse
        self.random_order = random_order
        self.random_degree = random_degree
        self.layers = nn.ModuleList()
        self.batch_norm = batch_norm
        self.residual = residual
        for lay in range(n_layer):
            self.layers.append(Cond_MAF_Layer(dim_in, dim_out, n_hidden, device, reverse, random_order, random_degree, residual))
            # print(lay)
            # if self.batch_norm and lay != (n_layer-1):
            if self.batch_norm:
                self.layers.append(BatchNormLayer(dim_in, dim_out))
                # self.layers.append(nn.BatchNorm1d(dim_out))

    def forward(self, x):
        log_det_sum = torch.zeros(x.shape[0], device=self.device)  # x.shape[0] is batch_size
        # layer_is_bn = False
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum += log_det
            '''
            if layer_is_bn:
                # x[:, (self.dim_in - self.dim_out):] = layer(x[:, (self.dim_in - self.dim_out):])
                bnout, log_det = layer(x[:, (self.dim_in - self.dim_out):])
                x = torch.cat((x[:, 0:(self.dim_in - self.dim_out)], bnout), dim=1)
                log_det_sum += log_det
            else:
                x, log_det = layer(x)
                log_det_sum += log_det
            layer_is_bn = not layer_is_bn
            '''
        return x, log_det_sum

    def backward(self, x):
        log_det_sum = torch.zeros(x.shape[0], device=self.device)
        for layer in reversed(self.layers):
            x, log_det = layer.backward(x)
            log_det_sum += log_det

        return x, log_det_sum

    def log_density_value_at_data(self, data_sample, theta_sample):
        x, log_det_sum = self.forward(torch.cat((data_sample, theta_sample), dim=1))
        u = x[:, (self.dim_in - self.dim_out):]
        log_density = - self.dim_out * torch.log(2 * torch.tensor(math.pi)) / 2 - (u ** 2).sum(dim=1) / 2 + log_det_sum
        return log_density

    def gen_sample(self, sample_size, x_0, qmc_flag=False, source=None):
        if source is None:
            dist = torch.distributions.MultivariateNormal(torch.zeros(self.dim_out), torch.diag(torch.ones(self.dim_out, )))
            normal_data = dist.sample((sample_size,)).to(self.device)
        else:
            normal_data = source
        input_data = torch.cat((x_0.repeat([normal_data.shape[0], 1]), normal_data), dim=1)
        out, _ = self.backward(input_data)
        # print('sample gene success')
        return out[:, (self.dim_in - self.dim_out):]


class Cond_NSF(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_hidden, device):
        """
        :param dim_in: dimension of (conditional) inputs
        :param dim_out: dimension of outputs
        :param n_layer: layer size of Block
        :param n_hidden: list of hidden units, default is [50, 50]
        :param device: -
        """

        super(Cond_NSF, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.nsf_from_sbibm = get_flow(model="nsf", dim_distribution=dim_out, dim_context=dim_in - dim_out,
                                       hidden_features=n_hidden[0], flow_num_transforms=n_layer).to(device)

    def log_density_value_at_data(self, data_sample, theta_sample):
        return self.nsf_from_sbibm.log_prob(theta_sample, data_sample)

    def gen_sample(self, sample_size, x_0, qmc_flag=False):
        return self.nsf_from_sbibm.sample(int(sample_size), x_0).squeeze(0)


class DefaultModelParam:
    def __init__(self):
        self.n_layer = 8  # 8
        self.batch_norm = False
        self.round = 20  # total round
        self.round_sample_size = 1000  # sample size for each round
        self.valid_rate = 0.05  # validation ratio
        self.valid_interval = 10
        self.eval_sample_size = 2000  # eval sample size
        self.medd_samp_size = self.eval_sample_size  # eval sample size
        self.medd_round = 1
        self.mmd_samp_size = self.eval_sample_size
        self.mmd_round = 1
        self.steps = 10000  # max steps
        self.print_state = self.steps
        self.print_state_time = 10000  # print time and valid loss
        self.batch_size = 100  # set batch size
        self.meshgrid_plot_point = 10
        self.figure_dpi = 400
        self.detected_log_file = False  # True: if file exist, exit the program
        self.plot_loss_figure_save = True
        self.plot_mmd_figure_save = False
        self.plot_theta_figure_save = False
        self.plot_density_figure_show = False
        self.plot_density_figure_save = False
        self.save_theta_csv = True
        self.clear_cuda_cache = True
        self.show_detailed_info = False
        self.linux_path = None
        self.manual_seed = None
        self.calc_c2st = True
