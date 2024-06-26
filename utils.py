import os
import shutil
import torch
import numpy as np
from matplotlib import pyplot as plt


def calib_kernel(x, x_0, rate):
    # return calibration kernel K(x, x_0)
    return torch.exp(-torch.sum((x - x_0) ** 2, dim=1) * rate ** 2 / 2)


def calib_kernel_ma(x, x_0, rate, invcov):
    # return calibration kernel K(x, x_0), Mahalanobis distance
    return torch.exp(-((x - x_0) @ invcov * (x - x_0)).sum(dim=1) * rate ** 2 / 2)


def clear_cache(c_FileSavePath):
    dir_list = ['output_loss', 'output_theta', 'output_log']
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


def init_args(args):
    # set torch device
    if args.gpu == 0:
        args.device = torch.device('cpu')
        print('using cpu')
    else:
        args.device = torch.device("cuda:" + str(args.gpu - 1))
        print('using gpu: %d' % args.gpu)
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("manual seed: " + str(args.seed))
    print("defensive_rate: " + str(args.dr))
    print("calib_kernel_rate: " + str(args.calib))
    print("mark str: " + args.mkstr)
    if args.dt == 0:
        print("density transformation is disabled.")
        args.density_transformation = False
    else:
        print("density transformation is enabled.")
        args.density_transformation = True
    if abs(args.ess) > 0.001:
        print("enable adaptive calib value (ESS).")
        args.enable_ess = True
        args.ess_alpha = args.ess
    else:
        print("disable adaptive calib value (ESS).")
        args.enable_ess = False
        args.ess_alpha = 0.
    if args.kn == 0:
        args.kernel_normalize = False
        print("disable kernel normalize.")
    else:
        args.kernel_normalize = True
        print("enable kernel normalize.")
    if args.upd == 0:
        args.proposal_update = False
    else:
        args.proposal_update = True
    if args.ear == 0:
        args.early_stop = False
        args.early_stop_tolarance = 0
        print("disable early stop.")
    else:
        args.early_stop = True
        args.early_stop_tolarance = args.ear
        print("enable early stop. torlarance: %d" % args.ear)
    if args.fl == 0:
        args.density_family_type = "MOG"
        print("conditional density type is mix of gaussian (MoG).")
    elif args.fl == 1:
        args.density_family_type = "NSF"
        print("conditional density type is neural spline flow (NSF).")
    else:
        raise NotImplementedError
    if args.clip > 1e-3:
        args.grad_clip = True
        print("using gradient clip at %.2f" % args.clip)
    else:
        args.grad_clip = False
    print("dbg1: %.6f, dbg2: %.6f" % (args.dbg1, args.dbg2))
