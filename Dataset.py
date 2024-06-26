# simulators for different datasets

import ctypes
import math
import os
import numpy as np
import pandas as pd
import torch

torch.multiprocessing.set_sharing_strategy('file_system')


class Simulator:
    def __init__(self, dataset_name, device, dtype=torch.float32, normalize=False):
        if os.name == 'posix':  # input File Save Path here
            self.FileSavePath = ""
        else:
            self.FileSavePath = ""
        self.dataset_name = dataset_name
        self.dim_theta = 0
        self.dim_x = 0
        self.device = device
        self.cache_theta = None
        self.dtype = dtype
        self.can_sample_from_post = False
        self.bounded_prior = False
        self.true_theta = None
        self.normalize = normalize
        assert dataset_name in ['two_moons', 'slcp', 'lotka', 'gandk', 'mg1', 'glu']
        if dataset_name == 'two_moons':
            self.dim_x = 2
            self.dim_theta = 2
            self.x_0 = torch.tensor([[0., 0.]], device=device)
            self.unif_lower = torch.tensor([-1., -1.], device=self.device)
            self.unif_upper = -self.unif_lower
            self.prior = torch.distributions.Independent(
                torch.distributions.Uniform(self.unif_lower, self.unif_upper, validate_args=False), 1)
            self.prior_type = 'unif'
            self.can_sample_from_post = True
            self.bounded_prior = True
            self.scale = torch.tensor([1.00, 1.00], device=self.device)
            self.true_theta = torch.tensor([[0.2475, 0.2475]], device=torch.device('cpu'))
            self.columns = ['$\\theta_1$', '$\\theta_2$']
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "output_abcr" + os.sep + "Da+0.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        elif dataset_name == 'slcp':
            self.dim_x = 8
            self.dim_theta = 5
            self.x_0 = torch.tensor([[1.4097, -1.8396, 0.8758, -4.4767, -0.1753, -3.1562, -0.6638, -2.7063]], device=device)
            self.unif_lower = torch.tensor([-3., -3., -3., -3., -3.], device=self.device)
            self.unif_upper = -self.unif_lower
            self.prior = torch.distributions.Independent(
                torch.distributions.Uniform(self.unif_lower, self.unif_upper, validate_args=False), 1)
            self.prior_type = 'unif'
            self.scale = torch.tensor([1.00, 0.81, 1.00, 0.81, 1.00, 0.81, 1.00, 0.81], device=self.device)
            self.can_sample_from_post = True
            self.bounded_prior = True
            self.true_theta = torch.tensor([[0.7, -2.9, -1., -0.9, 0.6]], device=torch.device('cpu'))
            self.columns = ['$\\theta_1$', '$\\theta_2$', '$\\theta_3$', '$\\theta_4$', '$\\theta_5$']
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "output_abcr" + os.sep + "Da+1.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        elif dataset_name == 'lotka':
            self.dim_x = 9
            self.dim_theta = 4
            self.unif_lower = torch.tensor([-5., -5., -5., -5.], device=self.device)
            self.unif_upper = torch.tensor([2., 2., 2., 2.], device=self.device)
            self.prior = torch.distributions.Independent(
                torch.distributions.Uniform(self.unif_lower, self.unif_upper, validate_args=False), 1)
            self.prior_type = 'unif'
            self.x_0 = torch.tensor([[4.6431, 4.0170, 7.1992, 6.6024, 0.9765, 0.9237, 0.9712, 0.9078, 0.047567]], device=self.device)
            self.scale = torch.tensor([0.3294, 0.5483, 0.6285, 0.9639, 0.0091, 0.0222, 0.0107, 0.0224, 0.1823], device=self.device)
            self.bounded_prior = True
            self.true_theta = torch.log(torch.tensor([[0.01, 0.5, 1, 0.01]], device=torch.device('cpu')))
            self.columns = ['$\\theta_1$', '$\\theta_2$', '$\\theta_3$', '$\\theta_4$']
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "output_abcr" + os.sep + "Da+2.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        elif dataset_name == 'gandk':
            self.dim_x = 4
            self.dim_theta = 4
            self.loc = torch.zeros(4).to(self.device)
            self.cov = torch.diag(torch.ones(4, ) * 4.).to(self.device)
            self.prior = torch.distributions.MultivariateNormal(self.loc, self.cov, validate_args=False)
            self.prior_type = 'normal'
            self.x_0 = torch.tensor([[2.9679, 1.5339, 0.4691, 1.7889]], device=device)
            self.scale = torch.tensor([0.0395, 0.1129, 0.0384, 0.1219], device=self.device)
            self.bounded_prior = False
            self.true_theta = torch.tensor([[3., 0., 2., 0.]], device=torch.device('cpu'))
            self.columns = ['$A$', '$\\log B$', '$g$', '$\\log(k+1/2)$']
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "output_abcr" + os.sep + "Da+3.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        elif dataset_name == 'mg1':
            self.dim_x = 5
            self.dim_theta = 3
            self.unif_lower = torch.tensor([0., 0., 0.], device=self.device)
            self.unif_upper = torch.tensor([10., 10., 1 / 3], device=self.device)
            self.prior = torch.distributions.Independent(
                torch.distributions.Uniform(self.unif_lower, self.unif_upper, validate_args=False), 1)
            self.prior_type = 'unif'
            self.x_0 = torch.log(torch.tensor([[1.0973, 2.3010, 4.2565, 7.2229, 23.3592]], device=self.device))
            self.scale = torch.tensor([0.1049, 0.1336, 0.1006, 0.1893, 0.2918], device=self.device)
            self.bounded_prior = True
            self.true_theta = torch.tensor([[1., 4., 0.2]], device=torch.device('cpu'))
            self.columns = ['$\\theta_1$', '$\\theta_2$', '$\\theta_3$']
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "output_abcr" + os.sep + "Da+4.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        elif dataset_name == 'glu':
            self.dim_x = 10
            self.dim_theta = self.dim_x
            self.unif_lower = torch.ones(self.dim_theta, device=self.device) * -1.
            self.unif_upper = torch.ones(self.dim_theta, device=self.device) * 1.
            self.prior = torch.distributions.Independent(
                torch.distributions.Uniform(self.unif_lower, self.unif_upper, validate_args=False), 1)
            self.prior_type = 'unif'
            self.x_0 = torch.tensor([[-0.5373, -0.2386, 0.8192, 0.6407, 0.4161,
                                      -0.0974, 1.1292, -0.0584, -0.9705, -0.9423]], device=self.device)
            self.scale = torch.ones(self.dim_theta, device=self.device) * np.sqrt(0.1)
            self.normal_dist = torch.distributions.MultivariateNormal(torch.zeros(self.dim_theta, device=self.device),
                                                                      0.1 * torch.diag(torch.ones(self.dim_theta, device=self.device)))
            self.bounded_prior = True
            self.true_theta = torch.tensor([[-0.9527, -0.1481, 0.9824, 0.4132, 0.9904,
                                             -0.7402, 0.7862, 0.0437, -0.6261, -0.7651]], device=torch.device('cpu'))
            self.columns = ['$\\theta_{' + str(i) + '}$' for i in range(1, self.dim_theta + 1)]
            self.reference_theta = torch.tensor(pd.read_csv(self.FileSavePath +
                                                            "output_abcr" + os.sep + "Da+5.csv").iloc[:, 1:(self.dim_theta + 1)].values,
                                                device=self.device, dtype=self.dtype)
        else:
            raise NotImplementedError
        self._x_0 = torch.clone(self.x_0)
        if self.normalize:
            self.x_0 = self.normalized_forward(self.x_0)

    def gen_data(self, para, param=None):
        # input param: theta
        # input shape: batch * dim_theta
        # output param: x from p(x|theta)
        # output shape: batch * dim_x
        batch = para.shape[0]
        if self.dataset_name == 'two_moons':
            unif_dist = torch.distributions.Uniform(- math.pi / 2, math.pi / 2)
            normal_dist = torch.distributions.Normal(0.1, 0.01)  # mu, sigma
            a_sample = unif_dist.sample((batch,)).to(self.device)  # batch * 1
            r_sample = normal_dist.sample((batch,)).to(self.device)  # batch * 1
            dim_1 = r_sample * torch.cos(a_sample) + 0.25 - (torch.abs(para[:, 0] + para[:, 1])) / torch.sqrt(torch.tensor(2.))
            dim_2 = r_sample * torch.sin(a_sample) + (-para[:, 0] + para[:, 1]) / torch.sqrt(torch.tensor(2.))
            return torch.stack((dim_1, dim_2), dim=1)
        elif self.dataset_name == 'slcp':
            normal_dist = torch.distributions.MultivariateNormal(torch.tensor([0., 0.], device=self.device),
                                                                 torch.diag(torch.ones(2, )).to(self.device))
            normal_sample = normal_dist.sample((batch, 4))
            mu = para[:, 0:2]
            sigma_x = torch.stack((
                para[:, 2] ** 2,
                (para[:, 3] ** 2) * torch.tanh(para[:, 4]),
                torch.zeros_like(para[:, 3]),
                # (para[:, 3] ** 2) * torch.sqrt(1. - torch.tanh(para[:, 4]) ** 2)
                (para[:, 3] ** 2) / torch.cosh(para[:, 4])
            ), dim=1).reshape(batch, 2, 2)
            normal_sample_2 = mu[:, None, :] + torch.bmm(normal_sample, sigma_x)
            return normal_sample_2.reshape(batch, -1)
        elif self.dataset_name == 'lotka':
            cpudv = torch.device('cpu')
            para_c = torch.exp(para.cpu().type(torch.float64))
            rand_int = torch.randint(65536, size=(batch,)).cpu().reshape(-1, 1)
            para_c = torch.cat((para_c, rand_int), dim=1)
            path_to_cfun = ""  # set .dll or .so file here for lotka model
            if os.sep == "\\":
                Cfun = ctypes.WinDLL(path_to_cfun + 'liblotka_c.dll', winmode=0)
            else:
                Cfun = ctypes.CDLL(path_to_cfun + 'liblotka_c.so')
            n = self.dim_x  # length for each task
            s = 14  # set multiprocess thread num here
            k = batch  # number of tasks
            input_value = torch.cat((para_c, torch.zeros(batch, n - 5).cpu()), dim=1)
            output_value = torch.zeros(input_value.shape[0], input_value.shape[1], dtype=self.dtype, device=cpudv)
            num_parts = (batch + k - 1) // k
            for i in range(num_parts):
                start_idx = i * k
                end_idx = min((i + 1) * k, batch)
                input_list = [float(s), float(k)] + input_value[start_idx:end_idx].reshape(-1).tolist()
                c_values = (ctypes.c_double * len(input_list))(*input_list)
                Cfun.lotka_multi_thread(c_values)
                output_value[start_idx:end_idx] = torch.tensor([c_values[j + 2] for j in range(len(c_values) - 2)], device=cpudv).reshape(-1, n)
            output_value[:, 0:2] = torch.log(output_value[:, 0:2] + 1.)
            return output_value.to(self.device)
        elif self.dataset_name == 'gandk':
            model_param = torch.clone(para)
            model_param[:, 3][model_param[:, 3] > 2] = 2
            unif_dist = torch.distributions.Uniform(0, 1)
            unif_samp = unif_dist.sample((batch, 1000))
            quantile_index = torch.linspace(1 / 8, 7 / 8, 7)
            quantile = torch.nanquantile(unif_samp, quantile_index, dim=1, interpolation='higher')  # 7 * batch
            normal_std_dist = torch.distributions.Normal(0., 1.)
            normal_quantile = normal_std_dist.icdf(quantile).to(self.device)  # 7 * batch
            # input para: A, log B, g, log(k+1/2)
            quantile_values = torch.zeros(batch, 7, device=self.device)
            for i in range(7):
                expmgz = torch.exp(-model_param[:, 2] * normal_quantile[i])
                quantile_values[:, i] = model_param[:, 0] + torch.exp(model_param[:, 1]) * (1 + 0.8 * (1 - expmgz) / (1 + expmgz)) * \
                                        torch.pow(1 + normal_quantile[i] ** 2, torch.exp(model_param[:, 3]) - 0.5) * normal_quantile[i]
            summary_1 = torch.stack((quantile_values[:, 3], quantile_values[:, 5] - quantile_values[:, 1]), dim=1)
            summary_2 = torch.stack((quantile_values[:, 5] + quantile_values[:, 1] - 2 * quantile_values[:, 3],
                                     quantile_values[:, 6] - quantile_values[:, 4] + quantile_values[:, 2] - quantile_values[:, 0]),
                                    dim=1) / summary_1[:, 1].repeat(2, 1).t()
            return torch.cat((summary_1, summary_2), dim=1)
            # return quantile_values
        elif self.dataset_name == 'mg1':
            # input para: theta_1, theta_2 - theta_1, theta_3
            job_num = 50
            quantile = torch.tensor([0., 0.25, 0.50, 0.75, 1.], device=self.device)
            zero_tensor = torch.tensor(0., device=self.device)
            unif_dist = torch.distributions.Uniform(0, 1)
            serv_time = para[:, 0].view(-1, 1) + unif_dist.sample((batch, job_num)).to(self.device) * para[:, 1].view(-1, 1)
            inter_time = -torch.log(unif_dist.sample((batch, job_num)).to(self.device) + 1e-8) / para[:, 2].view(-1, 1)
            arr_time = torch.cumsum(inter_time, dim=1)
            inter_left_time = torch.zeros(batch, job_num, device=self.device)
            left_time = torch.zeros(batch, job_num, device=self.device)
            inter_left_time[:, 0] = serv_time[:, 0] + arr_time[:, 0]
            left_time[:, 0] = inter_left_time[:, 0]
            for i in range(1, job_num):
                inter_left_time[:, i] = serv_time[:, i] + torch.max(zero_tensor, arr_time[:, i] - left_time[:, i - 1])
                left_time[:, i] = left_time[:, i - 1] + inter_left_time[:, i]
            return torch.log(torch.nanquantile(inter_left_time, quantile, dim=1).t())
        elif self.dataset_name == 'glu':
            return self.normal_dist.sample((batch,)) + para
        else:
            raise NotImplementedError
