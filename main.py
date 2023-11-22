# An SNPE-B Model test
# Ref: Lueckmann, Jan-Matthis, et al. "Flexible statistical inference for mechanistic models of neural dynamics."
# Advances in neural information processing systems 30 (2017).

import argparse
import copy
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import qmcpy as qp
import seaborn as sns
import shutil
import time
import torch
import torch.nn as nn
from sbibm import metrics

import Dataset
import SNPE_lib


def calib_kernel(x, x_0, rate):
    # return calibration kernel K(x, x_0)
    return torch.exp(-torch.sum((x - x_0) ** 2, dim=1) * rate ** 2 / 2)


def calib_kernel_ma(x, x_0, rate, invcov):
    # return calibration kernel K(x, x_0), Mahalanobis distance
    return torch.exp(-((x - x_0) @ invcov * (x - x_0)).sum(dim=1) * rate ** 2 / 2)


def clear_cache(c_output_density, c_output_loss, c_FileSavePath):
    if c_output_loss and c_output_density:
        dir_list = ['output_density', 'output_loss', 'output_theta', 'output_mmd', 'output_log']
        for name in dir_list:
            for root, dirs, files in os.walk(c_FileSavePath + os.sep + name):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
        print("delete cache success.")


def theta_forward_transform(theta, lower, upper):
    # transform theta from bounded regions(lower, upper) to unbounded regions(-inf, inf)
    return torch.log((theta - lower) / (upper - theta))


def theta_inverse_transform(theta, lower, upper):
    # transform theta from unbounded regions(-inf, inf) to bounded regions(lower, upper)
    return (upper * torch.exp(theta) + lower) / (torch.exp(theta) + 1)


def theta_forward_density(theta, lower, upper):
    # return log transformed density, theta range has no limited
    return torch.sum(torch.log(upper - lower) - torch.log(torch.tensor(2)) - torch.log(torch.cosh(theta) + 1), dim=1)


def theta_inverse_density(theta, lower, upper):
    # return log transformed density, theta range should in (lower, upper)
    return torch.sum(torch.log(upper - lower) - torch.log(upper - theta) - torch.log(theta - lower), dim=1)


def add_vline_in_plot(x, label, color):
    value = x.item()
    plt.axvline(value, color='red')


if __name__ == '__main__':
    # parse parameter
    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='gpu_available')  # 0: cpu; 1: cuda:0, 2: cuda:1, ...
    parser.add_argument('--dr', type=float, default=0.2, help='defensive_rate')  # 0.0: without defensive samping
    parser.add_argument('--calib', type=float, default=0.5, help='calibration para')  # 0.0: without calibration
    parser.add_argument('--reuse', type=int, default=1, help='sample_reuse')  # 0: no reuse; 1: type 1 loss(BH_weight); 2: type 2 loss(equal_weight)
    parser.add_argument('--mkstr', type=str, default="", help='mark_str')  # file name for identification
    parser.add_argument('--dt', type=int, default=0, help='density_transformation')  # 0: no density trans; 1: use density trans
    parser.add_argument('--ess', type=float, default=0.5, help='ess_alpha_value')  # 0: disable ESS
    parser.add_argument('--data', type=int, default=4, help='dataset')  # 0: two_moon; 1:slcp; 2:lotka; 3:g-and-k; 4:M/G/1
    parser.add_argument('--kn', type=int, default=1, help='kernel_normalize')  # 0: disable kernel normalize; 1: enable;
    parser.add_argument('--ear', type=int, default=20, help='early_stop')  # 0: disable early stop; N: early stop = N;
    parser.add_argument('--fl', type=int, default=2, help='flow_type')  # 0: mix of gaussian(MOG), 1: masked autoregressive flow(MAF) 2: NSF
    parser.add_argument('--clip', type=float, default=5.0, help='gradient_cut')  # clip the gradient
    parser.add_argument('--dbg1', type=int, default=1, help='debug_flag_1')
    parser.add_argument('--dbg2', type=int, default=50, help='debug_flag_2')
    args = parser.parse_args()
    if args.gpu == 0:
        print('using cpu')
        device = torch.device('cpu')
    else:
        print('using gpu: %d (in 1-4)' % args.gpu)
        device = torch.device("cuda:" + str(args.gpu - 1))
    defensive_rate = args.dr
    print("defensive_rate: " + str(defensive_rate))
    calib_kernel_rate = args.calib
    print("calib_kernel_rate: " + str(calib_kernel_rate))
    sample_reuse = args.reuse  # 0: no reuse; 1: type 1 loss; 2: type 2 loss
    mark_str = args.mkstr  # extra str for parallel running
    print("mark str: " + mark_str)
    if args.dt == 0:
        print("density transformation is disabled.")
        density_transformation = False
    else:
        print("density transformation is enabled.")
        density_transformation = True
    if abs(args.ess) > 1e-5:
        print("enable adaptive calib value (ESS).")
        enable_ess = True
        ess_alpha = args.ess
    else:
        print("disable adaptive calib value (ESS).")
        enable_ess = False
        ess_alpha = 0.
    dataset_arg = ['two_moons', 'slcp', 'lotka', 'gandk', 'mg1']
    print("using " + dataset_arg[args.data] + " dataset.")
    simulator = Dataset.Simulator(dataset_arg[args.data], device, torch.get_default_dtype())
    if args.kn == 0:
        kernel_normalize = False
        print("disable kernel normalize.")
    else:
        kernel_normalize = True
        print("enable kernel normalize.")
    if args.ear == 0:
        early_stop = False
        early_stop_tolarance = 0
        print("disable early stop.")
    else:
        early_stop = True
        early_stop_tolarance = args.ear
        print("enable early stop. torlarance: %d" % early_stop_tolarance)
    if args.fl == 0:
        density_family_type = "MOG"
        print("conditional density type is mix of gaussian.")
    elif args.fl == 1:
        density_family_type = "MAF"
        print("conditional density type is masked autoregressive flow.")
    elif args.fl == 2:
        density_family_type = "NSF"
        print("conditional density type is neural spline flow (NSF).")
    else:
        raise NotImplementedError
    if args.clip > 1e-3:
        grad_clip = True
        grad_clip_val = args.clip
        print("using gradient clip at %.2f" % args.clip)
    else:
        grad_clip = False
    print("dbg1: %.6f, dbg2: %.6f" % (args.dbg1, args.dbg2))
    plt.switch_backend("Agg")  # plt.switch_backend("tkagg")
    DefaultModelParam = SNPE_lib.DefaultModelParam()
    # input File Save Path here
    if os.sep == "/":
        FileSavePath = ""
    else:
        FileSavePath = ""
    print("File Save Path: " + FileSavePath)
    plot_loss_figure_show = False  # show loss figure
    plot_loss_figure_save = DefaultModelParam.plot_loss_figure_save  # save loss figure
    plot_mmd_figure_save = DefaultModelParam.plot_mmd_figure_save  # save mmd figure
    plot_theta_figure_save = DefaultModelParam.plot_theta_figure_save  # save theta figure
    plot_density_figure_show = False  # show density figure
    plot_density_figure_save = DefaultModelParam.plot_density_figure_save  # save density figure
    save_theta_csv = DefaultModelParam.save_theta_csv  # save theta csv
    clear_cuda_cache = DefaultModelParam.clear_cuda_cache  # clear cuda cache after each round
    model_save = True  # save model
    Mix_Gauss_keep_weight = False  # keep weight in mix of gaussian
    proposal_update = True  # update proposal
    calib_rate_update = False  # update calib_rate
    batch_norm = DefaultModelParam.batch_norm  # batch normalization
    pair_plot = False  # pair plot
    output_log = True  # save log file
    load_trained_proposal = False  # load trained proposal
    load_trained_model = False  # load trained model
    model_compile = True  # compile model (require torch 2.0)
    clear_output_density = False  # clear output dir
    clear_output_loss = clear_output_density
    debug_flag = False
    clear_cache(clear_output_density, clear_output_loss, FileSavePath)
    qmc_seed = 7  # QMC seed
    dim_x = simulator.dim_x
    dim_theta = simulator.dim_theta
    k = 8  # 8
    R = DefaultModelParam.round  # proposal update round, 20
    N = DefaultModelParam.round_sample_size  # sample generated size per round, 5000
    N_valid = int(DefaultModelParam.valid_rate * N)
    medd_samp_size = DefaultModelParam.medd_samp_size  # sample size use to evaluate median distance
    medd_round = DefaultModelParam.medd_round
    n_layer = DefaultModelParam.n_layer  # two-moons: 8, slcp: 16
    n_hidden = np.array([50, 50])  # 50, 50
    batch_size = DefaultModelParam.batch_size  # two-moons: 256, slcp: 1024
    steps = DefaultModelParam.steps  # 10000, steps in training network with N sample
    print_state = DefaultModelParam.print_state
    # print_state = 2500
    print_state_time = DefaultModelParam.print_state_time
    figure_dpi = DefaultModelParam.figure_dpi
    MMD_sample_size = DefaultModelParam.mmd_samp_size  # 1000
    MMD_round = DefaultModelParam.mmd_round  # 50
    if simulator.bounded_prior:
        transform_lower = simulator.unif_lower
        transform_upper = simulator.unif_upper
    if DefaultModelParam.manual_seed is not None:
        torch.manual_seed(DefaultModelParam.manual_seed)
    if model_compile and os.sep == '/':
        if density_family_type == "MAF":
            density_family_org = SNPE_lib.Cond_MAF(dim_x + dim_theta, dim_theta, n_layer, n_hidden, device, reverse=True,
                                                   batch_norm=batch_norm, random_order=False, random_degree=False, residual=False)
        elif density_family_type == "MOG":
            density_family_org = SNPE_lib.Cond_Mix_Gauss(dim_x, dim_theta, k, n_hidden, keep_weight=Mix_Gauss_keep_weight)
        elif density_family_type == "NSF":
            density_family_org = SNPE_lib.Cond_NSF(dim_x + dim_theta, dim_theta, n_layer, n_hidden, device)
        density_family = torch.compile(density_family_org, mode="max-autotune")
        print("using compiled model.")
        enable_model_compile = True
    else:
        if density_family_type == "MAF":
            density_family = SNPE_lib.Cond_MAF(dim_x + dim_theta, dim_theta, n_layer, n_hidden, device, reverse=True,
                                               batch_norm=batch_norm, random_order=False, random_degree=False, residual=False)
        elif density_family_type == "MOG":
            density_family = SNPE_lib.Cond_Mix_Gauss(dim_x, dim_theta, k, n_hidden, keep_weight=Mix_Gauss_keep_weight)
        elif density_family_type == "NSF":
            density_family = SNPE_lib.Cond_NSF(dim_x + dim_theta, dim_theta, n_layer, n_hidden, device)
        print("using uncompiled model.")
        enable_model_compile = False
    if device == torch.device('cuda:0') or device == torch.device('cuda:1') or device == torch.device('cuda:2') or device == torch.device('cuda:3'):
        density_family = density_family.to(device)
        torch.cuda.empty_cache()
    ModelInfo = "Mk+" + mark_str + "_Dr+" + ("%.2f" % defensive_rate) + "_Ca+" + ("%.2f" % calib_kernel_rate) + "_Es+" + ("%.2f" % ess_alpha) + \
                "_Kn+" + str(args.kn) + "_Re+" + str(sample_reuse) + "_Dt+" + str(int(density_transformation)) + "_Da+" + str(args.data) + \
                "_R+" + str(R) + "_N+" + str(N) + "_Ba+" + str(batch_size) + "_St+" + str(steps)
    if DefaultModelParam.detected_log_file:
        assert not os.path.exists(FileSavePath + 'output_log' + os.sep + 'log_' + ModelInfo + '.csv')
    batch_size += batch_size
    # X_0
    x_0 = simulator.x_0
    # Prior
    prior = simulator.prior
    # Proposal
    proposal = prior
    defensive_dist = prior
    mix_binomial = torch.distributions.Binomial(N, defensive_rate)
    # generate theta from proposal
    LossInfo = []
    LossInfo_valid = []
    LossInfo_x = 0
    LossInfo_x_list = []
    if output_log:
        output_log_idx = 0
        output_log_variables = {'round': int(),
                                'tau': float(),
                                'mcr': float(),
                                'alpha': float(),
                                'mmd': float(),
                                'nlog': float(),
                                'medd': float(),
                                'c2st': float(),
                                'iter': int(),
                                'mkstr': ''}
        output_log_df = pd.DataFrame(output_log_variables, index=[])
    optimizer = torch.optim.Adam(density_family.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-4)
    # store data for sample reuse
    full_theta = torch.tensor([], device=device)
    full_data = torch.tensor([], device=device)
    full_state_dict = []
    mmd_iter = []
    mmd_value = torch.tensor([])
    for r_idx in range(0, R):
        # theta sampling
        print("start theta sampling, round = " + str(r_idx))
        density_family.eval()
        with torch.no_grad():
            defensive_sample_size = torch.tensor(int(mix_binomial.sample()), device=device)
            proposal_sample_size = torch.tensor(N, device=device) - defensive_sample_size
            defensive_sample = defensive_dist.sample((defensive_sample_size,))
            if density_transformation:
                defensive_sample = theta_forward_transform(defensive_sample, transform_lower, transform_upper)
            if r_idx == 0 or (not proposal_update):
                # sampling theta from invariant proposal
                proposal_sample = proposal.sample((proposal_sample_size,))
                if density_transformation:
                    proposal_sample = theta_forward_transform(proposal_sample, transform_lower, transform_upper)
            else:
                # sampling theta from variant proposal
                with torch.no_grad():
                    # sampling theta from qF
                    proposal_sample = density_family.gen_sample(proposal_sample_size.item(), x_0, qmc_flag=False)
                    # resample if theta out of support
                    if (not density_transformation) and (not torch.all(prior.log_prob(proposal_sample) != float('-inf'))):
                        proposal_sample_in_support = proposal_sample[prior.log_prob(proposal_sample) != float('-inf')]
                        proposal_out_num = (proposal_sample_size - proposal_sample_in_support.shape[0]).item()
                        resample_times = 0
                        while True:
                            proposal_sample_extra = density_family.gen_sample(proposal_out_num * 3, x_0, qmc_flag=False)
                            proposal_sample_extra_in_support = proposal_sample_extra[prior.log_prob(proposal_sample_extra) != float('-inf')]
                            proposal_sample_in_support = torch.cat((proposal_sample_in_support, proposal_sample_extra_in_support), dim=0)
                            proposal_out_num = (proposal_sample_size - proposal_sample_in_support.shape[0]).item()
                            resample_times += 1
                            if proposal_out_num <= 0:
                                proposal_sample = proposal_sample_in_support[:proposal_sample_size]
                                break
                            if resample_times == 500:
                                print('proposal sampling error!')
                                break
                        # print('resample times: %d, out num: %d' % (resample_times, proposal_out_num))
                        assert torch.all(prior.log_prob(proposal_sample) != float('-inf'))
            theta_sample = torch.cat((proposal_sample, defensive_sample), dim=0)
            # shuffle
            theta_sample = theta_sample[torch.randperm(theta_sample.shape[0])]
            # data sampling
            time_start = time.perf_counter()
            if density_transformation:
                data_sample = simulator.gen_data(theta_inverse_transform(theta_sample, transform_lower, transform_upper))
            else:
                data_sample = simulator.gen_data(theta_sample)  # batch * dim_x
            if not torch.all(torch.logical_not(torch.isnan(data_sample))):
                print(data_sample[torch.isnan(data_sample)])
                print(theta_sample[torch.logical_not(torch.all(torch.logical_not(torch.isnan(data_sample)), dim=1))])
                raise ValueError
            time_end = time.perf_counter()
            print("%d data sampling time cost: %.2fs" % (theta_sample.shape[0], time_end - time_start))
            # sample reuse
            # print("start density calculating, round = " + str(r_idx))
            full_theta = torch.cat((full_theta, theta_sample), dim=0)
            full_data = torch.cat((full_data, data_sample), dim=0)
            if sample_reuse == 1 or sample_reuse == 2:
                perm = torch.randperm(full_theta.shape[0])
                theta_sample = full_theta[perm]  # theta_sample = full_theta
                data_sample = full_data[perm]  # data_sample = full_data
            # calculate log density
            dr_tensor = torch.tensor(defensive_rate, device=device)
            if density_transformation:
                defensive_log_density = defensive_dist.log_prob(theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                                        theta_forward_density(theta_sample, transform_lower, transform_upper)
            else:
                defensive_log_density = defensive_dist.log_prob(theta_sample)
            if r_idx == 0 or (not proposal_update):
                # if proposal not update, type 1 and type 2 loss will become origin loss
                if density_transformation:
                    proposal_log_density = proposal.log_prob(theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                                           theta_forward_density(theta_sample, transform_lower, transform_upper)
                else:
                    proposal_log_density = proposal.log_prob(theta_sample)
                theta_log_density = torch.logsumexp(torch.stack(
                    (defensive_log_density + torch.log(dr_tensor), proposal_log_density + torch.log(1 - dr_tensor)), dim=1), dim=1)
            elif sample_reuse == 0:
                with torch.no_grad():
                    proposal_log_density = density_family.log_density_value_at_data(x_0.repeat([N, 1]), theta_sample)
                theta_log_density = torch.logsumexp(torch.stack(
                    (defensive_log_density + torch.log(dr_tensor), proposal_log_density + torch.log(1 - dr_tensor)),
                    dim=1), dim=1)
            elif sample_reuse == 1:
                # flow proposal with sample reuse: TBA
                assert theta_sample.shape[0] == (r_idx + 1) * N
                full_log_density = (defensive_log_density + torch.log(dr_tensor)).reshape(-1, 1)  # ((r+1)*N) * 1
                ratio = torch.log(1 - dr_tensor) - torch.log(torch.tensor(r_idx + 1.))
                if density_transformation:
                    firs_log_density = proposal.log_prob(theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                                       theta_forward_density(theta_sample, transform_lower, transform_upper)
                else:
                    firs_log_density = proposal.log_prob(theta_sample)
                full_log_density = torch.cat((full_log_density, firs_log_density.reshape(-1, 1) + ratio), dim=1)
                prev_model = copy.deepcopy(density_family)
                for index in range(r_idx - 1):
                    prev_model.load_state_dict(full_state_dict[index])
                    prev_model.eval()
                    prev_log_density = prev_model.log_density_value_at_data(x_0.repeat([(r_idx + 1) * N, 1]), theta_sample)
                    full_log_density = torch.cat((full_log_density, prev_log_density.reshape(-1, 1) + ratio), dim=1)
                curr_log_density = density_family.log_density_value_at_data(x_0.repeat([(r_idx + 1) * N, 1]), theta_sample)
                full_log_density = torch.cat((full_log_density, curr_log_density.reshape(-1, 1) + ratio), dim=1)
                theta_log_density = torch.logsumexp(full_log_density, dim=1)
                full_state_dict.append(copy.deepcopy(density_family.state_dict()))
            elif sample_reuse == 2:
                assert theta_sample.shape[0] == (r_idx + 1) * N
                if density_transformation:
                    full_log_density = proposal.log_prob(theta_inverse_transform(theta_sample[0:N], transform_lower, transform_upper)) + \
                                       theta_forward_density(theta_sample[0:N], transform_lower, transform_upper)
                else:
                    full_log_density = proposal.log_prob(theta_sample[0:N])
                prev_model = copy.deepcopy(density_family)
                for index in range(r_idx - 1):
                    prev_model.load_state_dict(full_state_dict[index])
                    prev_model.eval()
                    prev_log_density = prev_model.log_density_value_at_data(x_0.repeat([N, 1]), theta_sample[((index + 1) * N):((index + 2) * N)])
                    full_log_density = torch.cat((full_log_density, prev_log_density), dim=0)
                curr_log_density = density_family.log_density_value_at_data(x_0.repeat([N, 1]), theta_sample[(r_idx * N):((r_idx + 1) * N)])
                full_log_density = torch.cat((full_log_density, curr_log_density), dim=0)
                theta_log_density = torch.logsumexp(
                    torch.stack((defensive_log_density + torch.log(dr_tensor), full_log_density + torch.log(1 - dr_tensor)), dim=1), dim=1)
                full_state_dict.append(copy.deepcopy(density_family.state_dict()))
            # calculate log prior
            if density_transformation:
                prior_log_prob = prior.log_prob(theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                                 theta_forward_density(theta_sample, transform_lower, transform_upper)
            else:
                prior_log_prob = prior.log_prob(theta_sample)
            # calculate ESS
            if kernel_normalize:
                kernel_inv_var = torch.inverse(torch.cov(data_sample.t()))
                data_calib = calib_kernel_ma(data_sample, x_0, calib_kernel_rate, invcov=kernel_inv_var)
            else:
                data_calib = calib_kernel(data_sample, x_0, calib_kernel_rate)
            ess_omega = data_calib * torch.exp(prior_log_prob - theta_log_density)
            ess_value = (torch.sum(ess_omega) ** 2) / (ess_omega.shape[0] * torch.sum(ess_omega ** 2))
            if enable_ess:  # calculate tau s.t. ess_value = ess_alpha, bisection
                if sample_reuse in [1, 2]:
                    if args.dbg1 == 1:  # (ess + log) / r_idx
                        ess_alpha_cur = ess_alpha * np.log(np.e + r_idx) / (1 + r_idx)
                    else:  # ess / r_idx
                        ess_alpha_cur = ess_alpha / (1 + r_idx)
                else:
                    ess_alpha_cur = ess_alpha
                # exp(-t): [0, inf) to [1, 0]
                rate_lower = 0.  # rate = +inf, small ess
                rate_upper = 1.  # rate = 0, ess = 1
                rate_scale = calib_kernel_rate / (np.log(4 / 3))  # e.g. rate_scale = 10.
                for ess_iter in range(50):
                    if (ess_value < ess_alpha_cur) or torch.isnan(ess_value).item():
                        rate_lower = np.exp(-calib_kernel_rate / rate_scale)
                    else:
                        rate_upper = np.exp(-calib_kernel_rate / rate_scale)
                    calib_kernel_rate = -rate_scale * np.log((rate_upper + rate_lower) / 2.)
                    if kernel_normalize:
                        data_calib = calib_kernel_ma(data_sample, x_0, calib_kernel_rate, invcov=kernel_inv_var)
                    else:
                        data_calib = calib_kernel(data_sample, x_0, calib_kernel_rate)
                    ess_omega = data_calib * torch.exp(prior_log_prob - theta_log_density)
                    ess_value = (torch.sum(ess_omega) ** 2) / (ess_omega.shape[0] * torch.sum(ess_omega ** 2))
                    # print("calib rate: %.4f, mean calib value: %.4f, ess value: %.4f." % (calib_kernel_rate, torch.mean(data_calib), ess_value))
            if False and os.sep == "\\":
                plt.hist(data_calib.cpu(), bins=100)
                plt.savefig(FileSavePath + 'output_theta' + os.sep + 'calib_' + ModelInfo + '_' + str(r_idx) + '.jpg', dpi=figure_dpi)
                plt.close()
            if calib_kernel_rate < 0.01:
                print('ESS alpha cannot be bigger!. set calib rate=%.4f' % calib_kernel_rate)
            print("updated calib rate: %.8f, mean calib value: %.8f, ess value: %.8f." % (calib_kernel_rate, torch.mean(data_calib), ess_value))
            if plot_theta_figure_save:
                # pair plot for theta
                if pair_plot and r_idx != 0:
                    if density_transformation:
                        plot_df = pd.DataFrame(theta_inverse_transform(density_family.gen_sample(N, x_0), transform_lower, transform_upper).cpu())
                        plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
                        g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
                        if simulator.true_theta is not None:
                            true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                            true_theta.columns = plot_df.columns
                            g.data = true_theta
                            g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                            g.map_diag(add_vline_in_plot)
                        plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta_' + ModelInfo + '_' +
                                    str(r_idx) + '.jpg', dpi=400)
                        plt.close()
                        plot_df = pd.DataFrame(density_family.gen_sample(N, x_0).cpu())
                        plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
                        g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
                        plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta(tr)_' + ModelInfo + '_' +
                                    str(r_idx) + '.jpg', dpi=400)
                        plt.close()
                    else:
                        plot_df = pd.DataFrame(density_family.gen_sample(N, x_0).cpu())
                        plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
                        g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
                        if simulator.true_theta is not None:
                            true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                            true_theta.columns = plot_df.columns
                            g.data = true_theta
                            g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                            g.map_diag(add_vline_in_plot)
                        plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta_' + ModelInfo + '_' +
                                    str(r_idx) + '.jpg', dpi=400)
                        plt.close()
        # network training
        caexp = data_calib * torch.exp(prior_log_prob - theta_log_density)  # exp of calib kernel rate
        valid_idx = data_sample.shape[0] - N_valid
        valid_data_sample = data_sample[valid_idx:]
        valid_theta_sample = theta_sample[valid_idx:]
        valid_prior_log_prob = prior_log_prob[valid_idx:]
        valid_theta_log_density = theta_log_density[valid_idx:]
        valid_data_calib = data_calib[valid_idx:]
        valid_caexp = valid_data_calib * torch.exp(valid_prior_log_prob - valid_theta_log_density)
        valid_loss_best = float('inf')
        valid_loss_best_idx = 0
        training_set = torch.utils.data.TensorDataset(data_sample[:valid_idx], theta_sample[:valid_idx], caexp[:valid_idx])
        dataset_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
        density_family.train()
        i = 0
        round_total_steps = steps
        while i < steps:
            for batch_data_sample, batch_theta_sample, batch_caexp in dataset_generator:
                loss = - torch.mean(density_family.log_density_value_at_data(batch_data_sample, batch_theta_sample) * batch_caexp)
                LossInfo.append(loss.detach().cpu().numpy())
                LossInfo_x += (1 / len(dataset_generator))
                LossInfo_x_list.append(LossInfo_x)
                optimizer.zero_grad()  # init gradient
                loss.backward()  # calculate gradient
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(density_family.parameters(), grad_clip_val)
                optimizer.step()  # update model parameters
            with torch.no_grad():
                # density_family.eval()
                valid_loss = - torch.mean(
                    density_family.log_density_value_at_data(valid_data_sample, valid_theta_sample) * valid_caexp).detach().cpu().numpy()
                LossInfo_valid.append(valid_loss)
                if valid_loss < valid_loss_best:
                    valid_loss_best = valid_loss
                    valid_loss_best_idx = i
                else:
                    if (i > (valid_loss_best_idx + early_stop_tolarance)) and early_stop:
                        round_total_steps = i + 1
                        i = steps - 1
                        print('round: %d, early stop condition satisfied.' % r_idx)
            if (i + 1) % print_state_time == 0:
                # print info
                print('----------')
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print('Newest Loss: %.4f, Validation Loss: %.4f' % (LossInfo[-1], LossInfo_valid[-1]))
                print('i: %d / %d, round: %d / %d, mkstr: %s' % ((i + 1), steps, r_idx, R - 1, mark_str))
            if (i + 1) % print_state == 0:
                # plot loss
                density_family.eval()
                if not density_transformation and DefaultModelParam.show_detailed_info:
                    print("inf num:", (torch.isinf(prior_log_prob).float().sum()))
                    print("inf rate:", (torch.isinf(prior.log_prob(theta_sample)).float().mean()))
                if plot_loss_figure_save:
                    plt.plot(LossInfo_x_list, LossInfo, '.', markersize=2)
                    plt.plot([loss_iter for loss_iter in range(len(LossInfo_valid))], LossInfo_valid, '.', markersize=2)
                    plt.xlabel("Number of iterations")
                    plt.ylabel("Loss")
                    plt.legend(['train loss', 'valid loss'])
                    plt.tight_layout()
                    plt.savefig(FileSavePath + 'output_loss' + os.sep + 'loss_' + ModelInfo + '_' +
                                str(r_idx) + '_' + str(i + 1) + '.jpg', dpi=figure_dpi)
                    plt.close()
                # calculate median distance
                medd = torch.tensor([0.])
                time_start = time.perf_counter()
                with torch.no_grad():
                    medd_result = torch.zeros(medd_round)
                    for medd_round_idx in range(medd_round):
                        medd_theta_samp = density_family.gen_sample(medd_samp_size, x_0)
                        if density_transformation:
                            medd_theta_samp = theta_inverse_transform(medd_theta_samp, transform_lower, transform_upper)
                        else:
                            medd_theta_samp = medd_theta_samp[prior.log_prob(medd_theta_samp) != float('-inf')]
                        medd_data_samp = simulator.gen_data(medd_theta_samp)
                        medd_result[medd_round_idx] = torch.nanmedian(torch.norm((medd_data_samp - x_0) / simulator.scale, dim=1)).cpu()
                    medd = torch.nanmedian(medd_result)
                time_end = time.perf_counter()
                time_medd = time_end - time_start
                # calculate c2st
                time_start = time.perf_counter()
                if DefaultModelParam.calc_c2st:
                    c2st = metrics.c2st(simulator.reference_theta.cpu(), medd_theta_samp.cpu())
                else:
                    c2st = torch.tensor([0.])
                time_end = time.perf_counter()
                time_c2st = time_end - time_start
                # calculate mmd
                time_start = time.perf_counter()
                mmd = metrics.mmd(simulator.reference_theta, medd_theta_samp)
                time_end = time.perf_counter()
                time_mmd = time_end - time_start
                # calculate negative log(true param)
                nlog = torch.tensor([0.])
                if simulator.true_theta is not None:
                    nlog_theta = simulator.true_theta.to(device)
                    if density_transformation:
                        with torch.no_grad():
                            nlog = -(density_family.log_density_value_at_data(
                                x_0, theta_forward_transform(nlog_theta, transform_lower, transform_upper)) +
                                     theta_inverse_density(nlog_theta, transform_lower, transform_upper)).cpu()
                    else:
                        with torch.no_grad():
                            nlog = -density_family.log_density_value_at_data(x_0, nlog_theta).cpu()
                print('medd: %.4f, time: %.2fs, c2st: %.4f, time: %.2fs, mmd: %.4f, time: %.2fs, nlog: %.4f, mkstr: %s' %
                      (medd.item(), time_medd, c2st.item(), time_c2st, mmd.item(), time_mmd, nlog.item(), mark_str))
                # save qF theta sample as csv file
                if save_theta_csv and r_idx == (R - 1):
                    plot_df = pd.DataFrame(medd_theta_samp.cpu())
                    plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
                    g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
                    if simulator.true_theta is not None:
                        true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                        true_theta.columns = plot_df.columns
                        g.data = true_theta
                        g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                        g.map_diag(add_vline_in_plot)
                    plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta_' + ModelInfo + '_' +
                                str(r_idx) + '_' + str(i + 1) + '.jpg', dpi=400)
                    plt.close()
                if save_theta_csv:
                    # using eval_sample
                    pd.DataFrame(medd_theta_samp.cpu()).to_csv(FileSavePath + 'output_theta' + os.sep + ModelInfo + '_' +
                                                               str(r_idx) + '.csv')
                # clear cache
                if device == torch.device('cuda:0') or device == torch.device('cuda:1') or device == torch.device('cuda:2') or device == torch.device(
                        'cuda:3'):
                    with torch.cuda.device(device):
                        if clear_cuda_cache:
                            torch.cuda.empty_cache()
                density_family.train()
            i += 1
        if output_log:
            # add metrics to log file
            output_log_df.loc[len(output_log_df.index)] = [r_idx + 1, calib_kernel_rate, torch.mean(data_calib).item(),
                                                           ess_value.item(),
                                                           mmd.item(), nlog.item(),
                                                           medd.item(), c2st.item(), round_total_steps, mark_str]
    # export log file and save model
    density_family.eval()
    with torch.no_grad():
        if plot_loss_figure_show:
            plt.figure()
            plt.plot(range(0, len(LossInfo), 1), LossInfo, '.', markersize=2)
            plt.show()
        if output_log:
            output_log_df.to_csv(FileSavePath + 'output_log' + os.sep + 'log_' + ModelInfo + '.csv')
        if model_save:
            if simulator.can_sample_from_post and False:
                mmd_iter = (torch.tensor(mmd_iter)).repeat_interleave(mmd_value.shape[1])
                mmd_info = torch.stack((mmd_iter, mmd_value.reshape(-1)))
                pd.DataFrame(mmd_info).to_csv(FileSavePath + 'output_mmd' + os.sep + 'mmd_' + ModelInfo + '.csv')
            pd.DataFrame(LossInfo).to_csv(FileSavePath + 'output_loss' + os.sep + 'loss_' + ModelInfo + '.csv')
            torch.save(density_family.state_dict(), FileSavePath + 'output_model' + os.sep + ModelInfo + ".pt")
