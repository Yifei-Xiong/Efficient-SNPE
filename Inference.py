# Description: This file is the main file for running the inference algorithm.

import torch
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import copy
import yaml
import seaborn as sns
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from sbibm import metrics
import SNPE_lib
import Dataset
import utils


def main(args, config):
    # init: set simulator
    device = args.device
    calib_kernel_rate = args.calib
    dataset_arg = ['two_moons', 'slcp', 'lotka', 'gandk', 'mg1', 'glu']
    print("using " + dataset_arg[args.data] + " dataset.")
    simulator = Dataset.Simulator(dataset_arg[args.data], device, torch.get_default_dtype())
    dim_x = simulator.dim_x
    dim_theta = simulator.dim_theta
    plt.switch_backend("Agg")  # plt.switch_backend("tkagg")
    FileSavePath = config['FileSavePath_linux'] if os.name == 'posix' else config['FileSavePath_win']
    print("File Save Path: " + FileSavePath)
    clear_output_folder = False  # clear loss, log, theta folder
    if clear_output_folder:
        utils.clear_cache(FileSavePath)
    model_compile = config['model_compile']
    R = config['rounds']  # Number of rounds
    N = config['round_sample_size']  # Number of samples for each round
    valid_rate = config['valid_rate']  # Validation rate
    eval_samp_size = config['eval_sample_size']  # Number of samples for evaluation
    n_layer = config['n_layer']  # Number of flow layers
    n_hidden = np.array(config['n_hidden'])  # Number of hidden units
    batch_size = config['batch_size']  # Batch size
    steps = config['max_steps_per_round']
    print_state = config['print_state']
    figure_dpi = config['figure_dpi']
    if config['save_log']:
        output_log_variables = {'round': int(),  # round index
                                'tau': float(),  # calib value
                                'mcr': float(),  # mean calib rate
                                'alpha': float(),  # ess value
                                'mmd': float(),  # mmd value
                                'nlog': float(),  # negative log likelihood
                                'lmd': float(),  # log median distance
                                'c2st': float(),  # c2st value
                                'iter': int(),  # training iteration
                                'mkstr': ''}  # mark string
        output_log_df = pd.DataFrame(output_log_variables, index=[])
    if simulator.bounded_prior:
        transform_lower = simulator.unif_lower
        transform_upper = simulator.unif_upper
    else:
        transform_lower = None
        transform_upper = None

    # model info
    ModelInfo = "Mk+" + args.mkstr + "_Dr+" + ("%.2f" % args.dr) + "_Ca+" + ("%.2f" % args.calib) + "_Es+" + ("%.2f" % args.ess_alpha) + \
                "_Re+" + str(args.reuse) + "_Dt+" + str(args.dt) + "_Da+" + str(args.data) + \
                "_R+" + str(R) + "_N+" + str(N) + "_Ba+" + str(batch_size) + "_St+" + str(steps)

    # check if log file exists
    if config['detected_log_file']:
        assert not os.path.exists(FileSavePath + 'output_log' + os.sep + 'log_' + ModelInfo + '.csv')

    # init: create conditional density model
    density_family = init_conditional_density(args, device, dim_theta, dim_x, model_compile, n_hidden, n_layer)
    # X_0
    x_0 = simulator.x_0
    # Prior
    prior = simulator.prior
    # Proposal
    proposal = prior
    defensive_dist = prior
    mix_binomial = torch.distributions.Binomial(N, args.dr)
    LossInfo = []
    LossInfo_valid = []
    LossInfo_x = 0
    LossInfo_x_list = []
    optimizer = torch.optim.Adam(density_family.parameters(), lr=1e-4, eps=1e-8, weight_decay=1e-4)
    # store data for sample reuse
    full_theta = torch.tensor([], device=device)
    full_data = torch.tensor([], device=device)
    full_state_dict = []
    # set MCMC parameters for SNL
    if args.method in ["SNL"]:
        args.mcmc_proposal_std = [0.2, 0.2, 0.6, 0.2, 0.2, 0.15][args.data]
        args.thin_mcmc_num = 10
        args.mcmc_init_value = prior.sample((1,))

    # training model
    for r_idx in range(0, R):
        if r_idx == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
        print("current lr: " + str(optimizer.param_groups[0]['lr']))

        # sampling theta
        print("start theta sampling, round = " + str(r_idx))
        defensive_sample_size = torch.tensor(int(mix_binomial.sample()), device=device)
        proposal_sample_size = torch.tensor(N, device=device) - defensive_sample_size
        defensive_sample = defensive_dist.sample((defensive_sample_size,))
        if args.density_transformation:
            defensive_sample = utils.theta_forward_transform(defensive_sample, transform_lower, transform_upper)
        if r_idx == 0 or (not args.proposal_update):
            # sampling theta from invariant proposal
            print("sampling theta from invariant proposal.")
            proposal_sample = proposal.sample((proposal_sample_size,))
            if args.density_transformation:
                proposal_sample = utils.theta_forward_transform(proposal_sample, transform_lower, transform_upper)
        else:
            # sampling theta from density_family
            proposal_sample = theta_sampling(args, density_family, prior, proposal_sample_size, x_0)
        theta_sample = torch.cat((proposal_sample, defensive_sample), dim=0)

        # sampling data
        time_start = time.perf_counter()
        if args.density_transformation:
            data_sample = simulator.gen_data(utils.theta_inverse_transform(theta_sample, transform_lower, transform_upper))
        else:
            data_sample = simulator.gen_data(theta_sample)  # batch * dim_x
        if not torch.all(torch.logical_not(torch.isnan(data_sample))):
            print(data_sample[torch.isnan(data_sample)])
            print(theta_sample[torch.logical_not(torch.all(torch.logical_not(torch.isnan(data_sample)), dim=1))])
            raise ValueError("dataset: %d, sample contains nan." % args.data)
        time_end = time.perf_counter()
        print("%d data sampling time cost: %.2fs" % (theta_sample.shape[0], time_end - time_start))

        # sample reuse
        full_theta = torch.cat((full_theta, theta_sample), dim=0)
        full_data = torch.cat((full_data, data_sample), dim=0)
        if (args.method == "SNPEB" and (args.reuse == 1 or args.reuse == 2)) or args.method in ["APT", "SNL"]:
            theta_sample = full_theta.clone()
            data_sample = full_data.clone()

        # calculate log proposal density
        if args.method in ["SNPEA", "APT", "SNL"]:
            if args.density_transformation:
                proposal_log_density = proposal.log_prob(utils.theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                                       utils.theta_forward_density(theta_sample, transform_lower, transform_upper)
            else:
                proposal_log_density = proposal.log_prob(theta_sample)
            theta_log_density = proposal_log_density
        if args.method == "SNPEB":
            theta_log_density = calculate_misr_log_density(args, N, r_idx, defensive_dist, proposal, full_state_dict, density_family, theta_sample,
                                                           transform_lower, transform_upper, x_0, device)

        # calculate log prior
        if args.density_transformation:
            prior_log_prob = prior.log_prob(utils.theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                             utils.theta_forward_density(theta_sample, transform_lower, transform_upper)
        else:
            prior_log_prob = prior.log_prob(theta_sample)

        # calculate ESS and calib value
        if args.enable_ess:  # calculate tau s.t. ess_value = ess_alpha, bisection
            calib_kernel_rate, data_calib, ess_value = calculate_data_calib(args, calib_kernel_rate, data_sample, prior_log_prob, r_idx,
                                                                            theta_log_density, x_0)
        else:
            # set data calib and ess to default value
            data_calib = torch.ones_like(prior_log_prob)
            ess_value = torch.tensor([1.])
        # plot calib value
        if False and os.name != 'posix':
            plt.hist(data_calib.cpu(), bins=100)
            plt.savefig(FileSavePath + 'output_theta' + os.sep + 'calib_' + ModelInfo + '_' + str(r_idx) + '.jpg', dpi=figure_dpi)
            plt.close()
        if calib_kernel_rate < 0.01:
            print('ESS alpha cannot be bigger!. set calib rate=%.4f' % calib_kernel_rate)
        print("updated calib rate: %.8f, mean calib value: %.8f, ess value: %.8f." % (calib_kernel_rate, torch.mean(data_calib), ess_value))

        # set dataloader
        valid_loss_best = float('inf')
        valid_loss_best_idx = 0
        dataset = torch.utils.data.TensorDataset(data_sample, theta_sample, theta_log_density, prior_log_prob, data_calib)
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(data_sample.shape[0] * (1 - valid_rate)),
                                                                               int(data_sample.shape[0] * valid_rate)])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        round_total_steps = steps
        range_generator = range(steps) if os.name == 'posix' else tqdm(range(steps))

        # network training
        for i in range_generator:
            # training model
            density_family.train()
            for batch_data, batch_theta, batch_log_density, batch_prior, batch_calib in train_loader:
                loss = torch.mean(loss_func(args, density_family, batch_data, batch_theta, batch_log_density, batch_prior, batch_calib))
                LossInfo.append(loss.detach().cpu().numpy())
                LossInfo_x += (1 / len(train_loader))
                LossInfo_x_list.append(LossInfo_x)
                optimizer.zero_grad()  # init gradient
                loss.backward()  # calculate gradient
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(density_family.parameters(), args.clip)
                optimizer.step()  # update model parameters

            # validation
            density_family.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_sample = 0
                for batch_data, batch_theta, batch_log_density, batch_prior, batch_calib in valid_loader:
                    valid_loss += torch.sum(loss_func(args, density_family, batch_data, batch_theta, batch_log_density, batch_prior, batch_calib))
                    valid_sample += batch_data.shape[0]
                valid_loss = valid_loss.detach().cpu().numpy() / valid_sample
                LossInfo_valid.append(valid_loss)
                if valid_loss < valid_loss_best:
                    valid_loss_best = valid_loss
                    valid_loss_best_idx = i
                    best_model_state_dict = copy.deepcopy(density_family.state_dict())
                else:
                    if (i > (valid_loss_best_idx + args.early_stop_tolarance)) and args.early_stop:
                        round_total_steps = i + 1
                        density_family.load_state_dict(best_model_state_dict)
                        print('round: %d, step: %d, early stop condition satisfied.' % (r_idx, round_total_steps))
                        break

            if (i + 1) % print_state == 0:
                # print info
                print('----------')
                now = datetime.datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print('Newest Loss: %.4f, Validation Loss: %.4f' % (LossInfo[-1], LossInfo_valid[-1]))
                print('i: %d / %d, round: %d / %d, mkstr: %s' % ((i + 1), steps, r_idx, R - 1, args.mkstr))

        # plot loss
        density_family.eval()
        if config['plot_loss_figure']:
            plt.plot(LossInfo_x_list, LossInfo, '.', markersize=2)
            plt.plot([loss_iter for loss_iter in range(len(LossInfo_valid))], LossInfo_valid, '.', markersize=2)
            plt.xlabel("Number of iterations")
            plt.ylabel("Loss")
            plt.legend(['train loss', 'valid loss'])
            plt.tight_layout()
            plt.savefig(FileSavePath + 'output_loss' + os.sep + 'loss_' + ModelInfo + '_' +
                        str(r_idx) + '_' + str(i + 1) + '.jpg', dpi=figure_dpi)
            plt.close()

        # generate eval theta sample
        eval_theta_samp = eval_theta_sampling(args, density_family, eval_samp_size, prior, transform_lower, transform_upper, x_0)
        lmd, c2st, mmd, nlog = calculate_metrics(args, config, eval_theta_samp, simulator, x_0)

        # save theta sample as csv file
        if config['save_theta_csv']:
            pd.DataFrame(eval_theta_samp.cpu()).to_csv(FileSavePath + 'output_theta' + os.sep + ModelInfo + '_final_' +
                                                       str(r_idx) + '.csv')

        # plot posterior theta
        if config['plot_theta_figure_each_round'] or (config['plot_theta_figure_last_round'] and r_idx == (R - 1)):
            plot_df = pd.DataFrame(eval_theta_samp.cpu())
            plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
            g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
            if simulator.true_theta is not None:
                true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
                true_theta.columns = plot_df.columns
                g.data = true_theta
                g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
                g.map_diag(utils.add_vline_in_plot)
            plt.savefig(FileSavePath + 'output_theta' + os.sep + 'theta_' + ModelInfo + '_' +
                        str(r_idx) + '.jpg', dpi=400)
            plt.close()
        if config['save_log']:
            output_log_df.loc[len(output_log_df.index)] = [r_idx + 1, calib_kernel_rate, torch.mean(data_calib).item(),
                                                           ess_value.item(),
                                                           mmd.item(), nlog.item(),
                                                           lmd.item(), c2st.item(), round_total_steps, args.mkstr]

    # save result
    if config['save_log']:
        output_log_df.to_csv(FileSavePath + 'output_log' + os.sep + 'log_' + ModelInfo + '.csv')
    if config['save_model']:
        pd.DataFrame(LossInfo).to_csv(FileSavePath + 'output_loss' + os.sep + 'loss_' + ModelInfo + '.csv')
        torch.save(density_family.state_dict(), FileSavePath + 'output_model' + os.sep + ModelInfo + ".pt")


def calculate_data_calib(args, calib_kernel_rate, data_sample, prior_log_prob, r_idx, theta_log_density, x_0):
    if args.kernel_normalize:
        kernel_inv_var = torch.inverse(torch.cov(data_sample.t()))
        data_calib = utils.calib_kernel_ma(data_sample, x_0, calib_kernel_rate, invcov=kernel_inv_var)
    else:
        data_calib = utils.calib_kernel(data_sample, x_0, calib_kernel_rate)
    if args.method == "SNPEB":
        density_ratio = torch.exp(prior_log_prob - theta_log_density)
    elif args.method == "APT":
        density_ratio = torch.ones_like(prior_log_prob)
    ess_omega = data_calib * density_ratio
    ess_value = (torch.sum(ess_omega) ** 2) / (ess_omega.shape[0] * torch.sum(ess_omega ** 2)).cpu()
    if args.reuse in [1, 2]:
        ess_alpha_cur = args.ess_alpha * (np.log(1 + r_idx) + 1) / (1 + r_idx)
    else:
        ess_alpha_cur = args.ess_alpha
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
        if args.kernel_normalize:
            data_calib = utils.calib_kernel_ma(data_sample, x_0, calib_kernel_rate, invcov=kernel_inv_var)
        else:
            data_calib = utils.calib_kernel(data_sample, x_0, calib_kernel_rate)
        ess_omega = data_calib * density_ratio
        ess_value = (torch.sum(ess_omega) ** 2) / (ess_omega.shape[0] * torch.sum(ess_omega ** 2)).cpu()
    return calib_kernel_rate, data_calib, ess_value


def calculate_misr_log_density(args, N, r_idx, defensive_dist, proposal, full_state_dict, density_family, theta_sample, transform_lower,
                               transform_upper, x_0, device):
    dr_tensor = torch.tensor(args.dr, device=device)
    if args.density_transformation:
        defensive_log_density = defensive_dist.log_prob(utils.theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                                utils.theta_forward_density(theta_sample, transform_lower, transform_upper)
    else:
        defensive_log_density = defensive_dist.log_prob(theta_sample)
    if r_idx == 0 or (not args.proposal_update):
        # if proposal not update, type 1 and type 2 loss will become origin loss
        if args.density_transformation:
            proposal_log_density = proposal.log_prob(utils.theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                                   utils.theta_forward_density(theta_sample, transform_lower, transform_upper)
        else:
            proposal_log_density = proposal.log_prob(theta_sample)
        theta_log_density = torch.logsumexp(torch.stack(
            (defensive_log_density + torch.log(dr_tensor), proposal_log_density + torch.log(1 - dr_tensor)), dim=1), dim=1)
    elif args.reuse == 0:
        with torch.no_grad():
            proposal_log_density = density_family.log_density_value_at_data(x_0.repeat([N, 1]), theta_sample)
        theta_log_density = torch.logsumexp(torch.stack(
            (defensive_log_density + torch.log(dr_tensor), proposal_log_density + torch.log(1 - dr_tensor)), dim=1), dim=1)
    elif args.reuse == 1:
        assert theta_sample.shape[0] == (r_idx + 1) * N
        full_log_density = (defensive_log_density + torch.log(dr_tensor)).reshape(-1, 1)  # ((r+1)*N) * 1
        ratio = torch.log(1 - dr_tensor) - torch.log(torch.tensor(r_idx + 1.))
        if args.density_transformation:
            firs_log_density = proposal.log_prob(utils.theta_inverse_transform(theta_sample, transform_lower, transform_upper)) + \
                               utils.theta_forward_density(theta_sample, transform_lower, transform_upper)
        else:
            firs_log_density = proposal.log_prob(theta_sample)
        full_log_density = torch.cat((full_log_density, firs_log_density.reshape(-1, 1) + ratio), dim=1)
        prev_model = copy.deepcopy(density_family)
        for index in range(r_idx - 1):
            prev_model.load_state_dict(full_state_dict[index])
            prev_model.eval()
            with torch.no_grad():
                prev_log_density = prev_model.log_density_value_at_data(x_0.repeat([(r_idx + 1) * N, 1]), theta_sample)
            full_log_density = torch.cat((full_log_density, prev_log_density.reshape(-1, 1) + ratio), dim=1)
        with torch.no_grad():
            curr_log_density = density_family.log_density_value_at_data(x_0.repeat([(r_idx + 1) * N, 1]), theta_sample)
        full_log_density = torch.cat((full_log_density, curr_log_density.reshape(-1, 1) + ratio), dim=1)
        theta_log_density = torch.logsumexp(full_log_density, dim=1)
        full_state_dict.append(copy.deepcopy(density_family.state_dict()))
    elif args.reuse == 2:
        assert theta_sample.shape[0] == (r_idx + 1) * N
        if args.density_transformation:
            full_log_density = proposal.log_prob(utils.theta_inverse_transform(theta_sample[0:N], transform_lower, transform_upper)) + \
                               utils.theta_forward_density(theta_sample[0:N], transform_lower, transform_upper)
        else:
            full_log_density = proposal.log_prob(theta_sample[0:N])
        prev_model = copy.deepcopy(density_family)
        for index in range(r_idx - 1):
            prev_model.load_state_dict(full_state_dict[index])
            prev_model.eval()
            with torch.no_grad():
                prev_log_density = prev_model.log_density_value_at_data(x_0.repeat([N, 1]), theta_sample[((index + 1) * N):((index + 2) * N)])
            full_log_density = torch.cat((full_log_density, prev_log_density), dim=0)
        with torch.no_grad():
            curr_log_density = density_family.log_density_value_at_data(x_0.repeat([N, 1]), theta_sample[(r_idx * N):((r_idx + 1) * N)])
        full_log_density = torch.cat((full_log_density, curr_log_density), dim=0)
        theta_log_density = torch.logsumexp(
            torch.stack((defensive_log_density + torch.log(dr_tensor), full_log_density + torch.log(1 - dr_tensor)), dim=1), dim=1)
        full_state_dict.append(copy.deepcopy(density_family.state_dict()))
    return theta_log_density


def eval_theta_sampling(args, density_family, eval_samp_size, prior, transform_lower, transform_upper, x_0):
    # generate eval theta sample
    if args.method in ["SNPEA", "SNPEB", "APT"]:
        with torch.no_grad():
            eval_theta_samp = density_family.gen_sample(eval_samp_size, x_0)
        if args.density_transformation:
            eval_theta_samp = utils.theta_inverse_transform(eval_theta_samp, transform_lower, transform_upper)
        else:
            eval_theta_samp = eval_theta_samp[prior.log_prob(eval_theta_samp) != float('-inf')]
    elif args.method in ["SNL"]:
        time_start = time.perf_counter()
        if args.snl_sampling_method == "MH":
            mcmc_log_density = lambda theta: density_family.log_density_value_at_data(theta, x_0.repeat([theta.shape[0], 1])) + prior.log_prob(theta)
            with torch.no_grad():
                eval_theta_samp = SNPE_lib.MCMC_MH(mcmc_log_density, args.mcmc_init_value,
                                                   sample_size=eval_samp_size, generate_size=int(eval_samp_size * args.thin_mcmc_num),
                                                   cut_size=eval_samp_size, proposal_std=args.mcmc_proposal_std, seq_sample=True, batch=50)
        elif args.snl_sampling_method == "Rejection":
            from sbi.samplers.rejection.rejection import rejection_sample
            def mcmc_log_density(theta):
                if len(theta.shape) == 1:
                    theta = theta.reshape(1, -1)
                return density_family.log_density_value_at_data(theta, x_0.repeat([theta.shape[0], 1])) + prior.log_prob(theta)

            eval_theta_samp, _ = rejection_sample(mcmc_log_density, prior, num_samples=eval_samp_size, show_progress_bars=False, device=args.device)
        else:
            raise ValueError("unknown SNL sampling method.")
        time_end = time.perf_counter()
        print("(eval) proposal_std = %.3f, mcmc generate time = %.3f s" % (args.mcmc_proposal_std, time_end - time_start))
    return eval_theta_samp


def theta_sampling(args, density_family, prior, proposal_sample_size, x_0):
    # generate proposal sample
    if args.method in ["SNPEA", "SNPEB", "APT"]:
        with torch.no_grad():
            proposal_sample = density_family.gen_sample(proposal_sample_size.item(), x_0)
            # resample if theta out of support
            if (not args.density_transformation) and (not torch.all(prior.log_prob(proposal_sample) != float('-inf'))):
                proposal_sample_in_support = proposal_sample[prior.log_prob(proposal_sample) != float('-inf')]
                proposal_out_num = (proposal_sample_size - proposal_sample_in_support.shape[0]).item()
                resample_times = 0
                while True:
                    proposal_sample_extra = density_family.gen_sample(proposal_out_num * 3 + 50, x_0)
                    proposal_sample_extra_in_support = proposal_sample_extra[prior.log_prob(proposal_sample_extra) != float('-inf')]
                    proposal_sample_in_support = torch.cat((proposal_sample_in_support, proposal_sample_extra_in_support), dim=0)
                    proposal_out_num = (proposal_sample_size - proposal_sample_in_support.shape[0]).item()
                    resample_times += 1
                    if proposal_out_num <= 0:
                        proposal_sample = proposal_sample_in_support[:proposal_sample_size]
                        break
                    if resample_times == 500:
                        print('proposal sampling error.')
                        break
                print('resample times: %d, out num: %d' % (resample_times, proposal_out_num))
                assert torch.all(prior.log_prob(proposal_sample) != float('-inf'))
    if args.method in ["SNL"]:
        # sampling theta from variant proposal
        time_start = time.perf_counter()
        if args.snl_sampling_method == "MH":
            mcmc_log_density = lambda theta: density_family.log_density_value_at_data(theta, x_0.repeat([theta.shape[0], 1])) + prior.log_prob(theta)
            with torch.no_grad():
                proposal_sample = SNPE_lib.MCMC_MH(mcmc_log_density, args.mcmc_init_value,
                                                   sample_size=proposal_sample_size, generate_size=int(proposal_sample_size * args.thin_mcmc_num),
                                                   cut_size=proposal_sample_size, proposal_std=args.mcmc_proposal_std, seq_sample=True, batch=50)
        elif args.snl_sampling_method == "Rejection":
            from sbi.samplers.rejection.rejection import rejection_sample
            def mcmc_log_density(theta):
                if len(theta.shape) == 1:
                    theta = theta.reshape(1, -1)
                return density_family.log_density_value_at_data(theta, x_0.repeat([theta.shape[0], 1])) + prior.log_prob(theta)

            proposal_sample, _ = rejection_sample(mcmc_log_density, prior, num_samples=proposal_sample_size.item(), show_progress_bars=False,
                                                  device=args.device)
        else:
            raise ValueError("unknown SNL sampling method.")
        time_end = time.perf_counter()
        print("proposal_std = %.3f, mcmc generate time = %.3f s" % (args.mcmc_proposal_std, time_end - time_start))
        args.mcmc_init_value = proposal_sample[-1].reshape(1, -1)
    return proposal_sample


def calculate_metrics(args, config, eval_theta_samp, simulator, x_0):
    # calculate log median distance (LMD)
    time_start = time.perf_counter()
    if config['calc_lmd']:
        with torch.no_grad():
            lmd_data_samp = simulator.gen_data(eval_theta_samp)
            lmd = torch.log(torch.nanmedian(torch.norm((lmd_data_samp - x_0) / simulator.scale, dim=1)).cpu())
    else:
        lmd = torch.tensor([0.])
    time_end = time.perf_counter()
    time_lmd = time_end - time_start

    # calculate classifier 2-sample tests (C2ST)
    time_start = time.perf_counter()
    if config['calc_c2st']:
        c2st = metrics.c2st(simulator.reference_theta.cpu(), eval_theta_samp.cpu())
    else:
        c2st = torch.tensor([0.])
    time_end = time.perf_counter()
    time_c2st = time_end - time_start

    # calculate maximum mean miscrepancy (MMD)
    time_start = time.perf_counter()
    if config['calc_mmd']:
        mmd = metrics.mmd(simulator.reference_theta, eval_theta_samp)
    else:
        mmd = torch.tensor([0.])
    time_end = time.perf_counter()
    time_mmd = time_end - time_start

    # calculate negative log likelihood (NLOG)
    time_start = time.perf_counter()
    if config['calc_nlog']:
        kde = KernelDensity(bandwidth="scott", kernel='gaussian').fit(eval_theta_samp.cpu())
        nlog = -kde.score_samples(simulator.true_theta.cpu())
    else:
        nlog = torch.tensor([0.])
    time_end = time.perf_counter()
    time_nlog = time_end - time_start
    print('lmd: %.4f, time: %.2fs, c2st: %.4f, time: %.2fs, mmd: %.4f, time: %.2fs, nlog: %.4f, time: %.2fs, mkstr: %s' %
          (lmd.item(), time_lmd, c2st.item(), time_c2st, mmd.item(), time_mmd, nlog.item(), time_nlog, args.mkstr))
    return lmd, c2st, mmd, nlog


def init_conditional_density(args, device, dim_theta, dim_x, model_compile, n_hidden, n_layer):
    # init conditional density model
    if args.method in ["SNPEA", "SNPEB", "APT"]:
        if model_compile and os.name == 'posix':
            if args.density_family_type == "MOG":
                density_family_org = SNPE_lib.Cond_Mix_Gauss(dim_x, dim_theta, n_hidden, keep_weight=False)
            elif args.density_family_type == "NSF":
                density_family_org = SNPE_lib.Cond_NSF(dim_x + dim_theta, dim_theta, n_layer, n_hidden, device)
            density_family = torch.compile(density_family_org, mode="max-autotune")
            print("using compiled model.")
        else:
            if args.density_family_type == "MOG":
                density_family = SNPE_lib.Cond_Mix_Gauss(dim_x, dim_theta, n_hidden, keep_weight=False)
            elif args.density_family_type == "NSF":
                density_family = SNPE_lib.Cond_NSF(dim_x + dim_theta, dim_theta, n_layer, n_hidden, device)
            print("using uncompiled model.")
        if device == torch.device('cuda:0') or device == torch.device('cuda:1') or device == torch.device('cuda:2') or device == torch.device(
                'cuda:3'):
            density_family = density_family.to(device)
        return density_family
    elif args.method in ["SNL"]:
        if model_compile and os.name == 'posix':
            if args.density_family_type == "MOG":
                density_family_org = SNPE_lib.Cond_Mix_Gauss(dim_theta, dim_x, n_hidden, keep_weight=False)
            elif args.density_family_type == "NSF":
                density_family_org = SNPE_lib.Cond_NSF(dim_x + dim_theta, dim_x, n_layer, n_hidden, device)
            density_family = torch.compile(density_family_org, mode="max-autotune")
            print("using compiled model.")
        else:
            if args.density_family_type == "MOG":
                density_family = SNPE_lib.Cond_Mix_Gauss(dim_theta, dim_x, n_hidden, keep_weight=False)
            elif args.density_family_type == "NSF":
                density_family = SNPE_lib.Cond_NSF(dim_x + dim_theta, dim_x, n_layer, n_hidden, device)
            print("using uncompiled model.")
        if device != torch.device('cpu'):
            density_family = density_family.to(device)
        return density_family


def loss_func(args, density_family, batch_data, batch_theta, batch_log_density, batch_prior, batch_calib):
    # calculate loss
    if args.method == "SNPEA":
        loss = - density_family.log_density_value_at_data(batch_data, batch_theta)
    elif args.method == "SNPEB":
        loss = - density_family.log_density_value_at_data(batch_data, batch_theta) * torch.exp(batch_prior - batch_log_density) * batch_calib
    elif args.method == "APT":
        atoms = args.atoms
        bs = batch_data.shape[0]
        probs = torch.ones(bs, bs) * (1 - torch.eye(bs)) / (bs - 1)
        inner_theta_idx = torch.multinomial(probs, num_samples=atoms - 1, replacement=False)
        inner_theta = torch.cat((batch_theta[:, None, :], batch_theta[inner_theta_idx]), dim=1).reshape(bs * atoms, -1)
        inner_prior = torch.cat((batch_prior[:, None], batch_prior[inner_theta_idx]), dim=1)
        data_expand = batch_data.view(bs, 1, -1).expand(bs, atoms, -1).reshape(bs * atoms, -1)
        log_prob = density_family.log_density_value_at_data(data_expand, inner_theta).view(bs, atoms) - inner_prior
        loss = (-log_prob[:, 0] - batch_log_density + torch.logsumexp(log_prob, dim=1) - np.log(atoms)) * batch_calib
    elif args.method == "SNL":
        loss = - density_family.log_density_value_at_data(batch_theta, batch_data)
    return loss


if __name__ == '__main__':
    # parse parameter
    default_dtype = torch.float32
    torch.set_default_dtype(default_dtype)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="config file path")
    parser.add_argument('--gpu', type=int, default=1, help='gpu_available')  # 0: cpu; 1: cuda:0, 2: cuda:1, ...
    parser.add_argument('--method', type=str, default='SNL', help='method name')  # method should be: SNPEA, SNPEB, APT, SNL
    parser.add_argument('--data', type=int, default=1, help='dataset')  # 0: two_moon; 1:slcp; 2:lotka; 3:g-and-k; 4:M/G/1; 5: glu
    parser.add_argument('--seed', type=int, default=10019, help='set manual seed')
    parser.add_argument('--calib', type=float, default=0.0, help='set calibration value')  # 0.0: without calibration
    parser.add_argument('--dr', type=float, default=0.2, help='set defensive rate')  # 0.0: without defensive samping
    parser.add_argument('--reuse', type=int, default=0, help='sample reuse type')  # 0: no reuse; 1: type 1 loss; 2: type 2 loss
    parser.add_argument('--dt', type=int, default=1, help='enable density transformation')  # 0: no density trans; 1: use density trans
    parser.add_argument('--ess', type=float, default=0.0, help='set ess value')  # 0: disable ESS
    parser.add_argument('--kn', type=int, default=1, help='enable kernel normalize')  # 0: disable kernel normalize; 1: enable kernel normalize;
    parser.add_argument('--atoms', type=int, default=10, help='set apt atoms number')
    parser.add_argument('--upd', type=int, default=1, help='proposal update')  # 0: disable proposal update; 1: enable proposal update;
    parser.add_argument('--ear', type=int, default=20, help='enable earlystop')  # 0: disable early stop; N: early stop torlarance = N;
    parser.add_argument('--fl', type=int, default=1, help='set flow type')  # 0: mix of gaussian(MOG), 1: NSF
    parser.add_argument('--clip', type=float, default=5.0, help='enable gradient cut')
    parser.add_argument('--mkstr', type=str, default="i119", help='set markstr')
    parser.add_argument('--dbg1', type=int, default=1, help='debug flag 1')
    parser.add_argument('--dbg2', type=int, default=50, help='debug flag 2')
    args = parser.parse_args()
    if args.method == "SNPEA":
        args.fl = 0
        args.ess = 0
        args.dr = 0.0
        args.upd = 0
    elif args.method == "SNPEB":
        args.fl = 1
    elif args.method == "APT":
        args.fl = 1
        args.dr = 0.0
    elif args.method == "SNL":
        args.fl = 1
        args.dt = 0
        args.ess = 0
        args.dr = 0.0
        args.snl_sampling_method = "Rejection"  # "MH" or "Rejection" sampling from likelihood estimator
        assert args.snl_sampling_method in ["MH", "Rejection"]
    else:
        raise NotImplementedError
    utils.init_args(args)
    with open(args.config) as file:
        config = yaml.safe_load(file)
    main(args, config)
