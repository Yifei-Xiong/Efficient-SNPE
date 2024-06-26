# Test SMC-ABC methods on likelihood-free problems
# Code reference: https://github.com/gpapamak/epsilon_free_inference

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import seaborn as sns
import yaml
import Dataset


def calc_dist(stats_1, stats_2):
    """Calculates the distance between two observations. Here the euclidean distance is used."""
    diff = stats_1 - stats_2
    dist = np.sqrt(np.dot(diff, diff))
    return dist


def calc_dist_batch(stats_1, stats_2, scale=None):
    """Calculates the distance between two observations. Here the euclidean distance is used."""
    # stats_1: shape: batch * dim
    # state_2: shape: batch * dim
    if scale is None:
        return torch.sqrt(torch.sum((stats_1 - stats_2) ** 2, dim=1))
    else:
        return torch.sqrt(torch.sum(((stats_1 - stats_2) / scale) ** 2, dim=1))


def add_vline_in_plot(x, label, color):
    value = x.item()
    plt.axvline(value, color='red')


def run_smc_abc(simulator, n_particles, batch, eps_init, eps_last, eps_decay, ess_min, param):
    """Runs SMC-ABC and saves results."""
    # set parameters
    round_exc = np.log(eps_last / eps_init) / np.log(eps_decay) + 1
    print("batch size: %d" % batch)
    print("start eps: %s" % eps_init)
    print("exit eps: %s" % eps_last)
    print("round: %.2f" % round_exc)
    ModelInfo, FileSavePath = param
    # load observed data
    obs_stats = simulator.x_0
    n_dim = simulator.dim_theta
    all_ps = []
    all_logweights = []
    all_eps = []
    all_nsims = []
    # sample initial population
    ps = torch.zeros(n_particles, n_dim)
    weights = torch.ones(n_particles) / n_particles
    logweights = torch.log(weights)
    eps = eps_init
    iter_times = 0
    nsims = 0
    # scale = simulator.scale
    scale = None
    prior = simulator.prior
    batch_obs_states = obs_stats.repeat(batch, 1)
    generated_samp = 0

    # first round: sampling from prior
    while True:
        batch_sample = prior.sample((batch,))  # batch * dim_theta
        batch_data = simulator.gen_data(batch_sample)  # batch * dim_data
        dist = calc_dist_batch(batch_data, batch_obs_states, scale)  # batch
        new_sample = batch_sample[dist < eps]
        if new_sample.shape[0] == 0:
            nsims += batch
            continue
        else:
            if generated_samp + new_sample.shape[0] >= n_particles:
                sample_part_size = n_particles - generated_samp
                ps[generated_samp:] = new_sample[:sample_part_size]
                nsims += (((dist < eps) == True).nonzero()[sample_part_size - 1]).item()
                break
            else:
                ps[generated_samp:(generated_samp + new_sample.shape[0])] = new_sample
                generated_samp += new_sample.shape[0]
                nsims += batch
                print("nsims at first round: %d, generated sample: %d" % (nsims, generated_samp))
                continue
    all_ps.append(ps.clone())
    all_logweights.append(logweights.clone())
    all_eps.append(eps)
    all_nsims.append(nsims)
    print('iteration = %d, eps = %.4f, ess = %.4f, sim_num = %d' % (iter_times, eps, 1.0, nsims))

    # second and later rounds
    while True:

        # save csv and plot
        save_csv_and_plot(FileSavePath, ModelInfo, iter_times, ps, simulator, save=False, plot=False)
        if eps <= eps_last:
            break

        # calculate population covariance
        iter_times += 1
        eps *= eps_decay
        mean = torch.mean(ps, dim=0)
        cov = 2.0 * (ps.t() @ ps / n_particles - torch.outer(mean, mean))
        std = torch.linalg.cholesky(cov)

        # perturb particles
        new_ps = torch.zeros_like(ps)
        new_logweights = torch.zeros_like(logweights)
        discrete_sampler = torch.distributions.Categorical(weights)
        normal_sampler = torch.distributions.Normal(0., 1.)
        generated_samp = 0

        # sample from particles
        while True:
            batch_idx = discrete_sampler.sample((batch,))  # batch
            normal_sample = normal_sampler.sample((n_dim, batch))
            batch_new_ps = ps[batch_idx] + (std @ normal_sample).t()  # batch * dim_theta
            batch_data = simulator.gen_data(batch_new_ps)  # batch * dim_data
            dist = calc_dist_batch(batch_data, batch_obs_states, scale)  # batch
            prop_idx = torch.logical_and(dist < eps, prior.log_prob(batch_new_ps) != float('-inf'))
            new_sample = batch_new_ps[prop_idx]
            if new_sample.shape[0] == 0:
                nsims += batch
                continue
            else:
                if generated_samp + new_sample.shape[0] >= n_particles:
                    sample_part_size = n_particles - generated_samp
                    new_ps[generated_samp:] = new_sample[:sample_part_size]
                    nsims += (prop_idx.nonzero()[sample_part_size - 1]).item()
                    break
                else:
                    new_ps[generated_samp:(generated_samp + new_sample.shape[0])] = new_sample
                    generated_samp += new_sample.shape[0]
                    nsims += batch
                    print("nsims at %d/%d round: %d, generated sample: %d" % (iter_times, round_exc, nsims, generated_samp))
                    continue
        for i in range(n_particles):
            logkernel = -0.5 * torch.sum(torch.linalg.solve(std, (new_ps[i] - ps).t()) ** 2, dim=0)
            new_logweights[i] = float('-inf') if prior.log_prob(new_ps[i]).item() == float('-inf') else -torch.logsumexp(logweights + logkernel,
                                                                                                                         dim=0)
        ps = new_ps
        logweights = new_logweights - torch.logsumexp(new_logweights, dim=0)
        weights = torch.exp(logweights)

        # calculate effective sample size
        ess = 1.0 / (torch.sum(weights ** 2) * n_particles)
        print('iteration = %d, eps = %.4f, ess = %.4f, sim_num = %d' % (iter_times, eps, ess, nsims))
        if ess < ess_min:
            # resample particles
            discrete_sampler = torch.distributions.Categorical(weights)
            idx = discrete_sampler.sample((n_particles,))
            ps = ps[idx]
            weights = torch.ones(n_particles) / n_particles
            logweights = torch.log(weights)
        all_ps.append(ps.clone())
        all_logweights.append(logweights.clone())
        all_eps.append(eps)
        all_nsims.append(nsims)
    return all_ps, all_logweights, all_eps, all_nsims


def save_csv_and_plot(FileSavePath, ModelInfo, iter_times, ps, simulator, save=True, plot=True):
    plot_df = pd.DataFrame(ps.cpu())
    if save:
        plot_df.to_csv(FileSavePath + 'output_abc' + os.sep + 'theta_' + ModelInfo + '_' + str(iter_times) + '.csv')
    if plot:
        plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
        g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
        if simulator.true_theta is not None:
            true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
            true_theta.columns = plot_df.columns
            g.data = true_theta
            g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
            g.map_diag(add_vline_in_plot)
        plt.savefig(FileSavePath + 'output_abc' + os.sep + 'theta_' + ModelInfo + '_' + str(iter_times) + '.jpg', dpi=400)
        plt.close()


def main(args, config):
    # init
    if args.gpu == 0:
        print('using cpu')
        device = torch.device('cpu')
    else:
        print('using gpu: %d' % args.gpu)
        device = torch.device("cuda:" + str(args.gpu - 1))
    torch.set_default_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("manual seed: " + str(args.seed))
    print("mark str: " + args.mkstr)
    dataset_arg = ['two_moons', 'slcp', 'lotka', 'gandk', 'mg1', 'glu']
    print("using " + dataset_arg[args.data] + " dataset.")
    simulator = Dataset.Simulator(dataset_arg[args.data], device, torch.get_default_dtype())
    plt.switch_backend("Agg")
    FileSavePath = config['FileSavePath_linux'] if os.name == 'posix' else config['FileSavePath_win']
    print("File Save Path: " + FileSavePath)
    ModelInfo = "Mk+ABC_" + args.mkstr + "_Da+" + str(args.data)

    # run SMC-ABC
    all_ps, all_logweights, all_eps, all_nsims = run_smc_abc(simulator, args.particles, args.bs, args.eps_start,
                                                             args.eps_end, args.eps_decay, args.ess, (ModelInfo, FileSavePath))

    # save results
    all_nsims_df = pd.DataFrame(all_nsims)
    all_nsims_df.to_csv(FileSavePath + 'output_abc' + os.sep + 'nsims_' + ModelInfo + '.csv', index=False)
    for ps, logweights, eps, idx in zip(all_ps, all_logweights, all_eps, range(len(all_ps))):
        if idx != (len(all_ps) - 1):
            continue
        plot_df = pd.DataFrame(ps.cpu())
        plot_df.to_csv(FileSavePath + 'output_abc' + os.sep + 'theta_' + ModelInfo + '.csv')
        plot_df.columns = simulator.columns if (simulator.columns is not None) else plot_df.columns
        g = sns.pairplot(plot_df, plot_kws=dict(marker="+", s=0.2, linewidth=1))
        if simulator.true_theta is not None:
            true_theta = pd.DataFrame(simulator.true_theta.detach().numpy())
            true_theta.columns = plot_df.columns
            g.data = true_theta
            g.map_offdiag(sns.scatterplot, s=120, marker=".", edgecolor="black")
            g.map_diag(add_vline_in_plot)
        plt.savefig(FileSavePath + 'output_abc' + os.sep + 'theta_' + ModelInfo + '.jpg', dpi=400)
        plt.close()


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="config file path")
    parser.add_argument('--gpu', type=int, default=4, help='gpu_available')  # 0: cpu; 1: cuda:0, 2: cuda:1, ...
    parser.add_argument('--mkstr', type=str, default="abcr", help='set markstr')  # mark string
    parser.add_argument('--bs', type=int, default=0, help='batch size')  # batch size
    parser.add_argument('--seed', type=int, default=10000, help='set manual seed')
    parser.add_argument('--particles', type=int, default=2000, help='number of particles')  # number of particles
    parser.add_argument('--eps_start', type=float, default=0.0, help='initial epsilon')  # initial epsilon
    parser.add_argument('--eps_end', type=float, default=0.0, help='final epsilon')  # final epsilon
    parser.add_argument('--eps_decay', type=float, default=0.8, help='epsilon decay rate')  # epsilon decay rate
    parser.add_argument('--ess', type=float, default=0.5, help='ess threshold')  # effective sample size threshold
    parser.add_argument('--data', type=int, default=4, help='dataset')  # 0: two_moon; 1:slcp; 2:lotka; 3:g-and-k; 4:M/G/1; 5: glu
    args = parser.parse_args()
    with open(args.config) as file:
        config = yaml.safe_load(file)
    if args.bs == 0:
        # default batch size (for Nvidia 2080Ti with 11GB memory)
        args.bs = [50000000, 30000000, 100000, 300000, 5000000, 20000000][args.data]
    if args.eps_start == 0.0:
        # default initial epsilon
        args.eps_start = [0.05, 10.0, 2.0, 0.25, 0.30, 3.0][args.data]
    if args.eps_end == 0.0:
        args.eps_end = args.eps_start / 10.0
    main(args, config)
