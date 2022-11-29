import torch
import math
import numpy as np

def log10(x):
      """Convert a new tensor with the base-10 logarithm of the elements of x. """
      return torch.log(x+1e-6) / math.log(10)

class Result(object):
    def __init__(self):
        self.mse, self.rmse, self.mae, self.rmselg10 = 0, 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def set_to_worst(self):
        # self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae, self.rmselg10 = np.inf, np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    # def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3):
    def update(self, mse, rmse, mae, rmselg10, absrel, lg10, delta1, delta2, delta3):
        # self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae, self.rmselg10 = mse, rmse, mae, rmselg10
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3


    def evaluate(self, output, target):
        output[output == 0] = 0.00001
        target[target == 0] = 0.00001
        valid_mask = ((target>0) + (output>0)) > 0

        # output = 1e3 * output[valid_mask]
        # target = 1e3 * target[valid_mask]
        output = output[valid_mask]
        target = target[valid_mask]
        abs_diff = np.abs(output - target)
        # abs_diff = (output - target).abs()

        self.mse = np.mean(abs_diff ** 2)
        self.rmse = np.sqrt((abs_diff ** 2).mean())
        self.mae = np.mean(abs_diff)

        # self.mse = float((torch.pow(abs_diff, 2)).mean())
        # self.rmse = math.sqrt(self.mse)
        # self.mae = float(abs_diff.mean())
        err = np.abs(np.log10(output) - np.log10(target))

        self.rmselg10 = np.sqrt(np.mean(err ** 2))

        self.lg10 = np.mean(err)
        # self.absrel = np.mean(abs_diff / target)

        # self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = np.maximum(output / target, target / output)
        self.delta1 = (maxRatio < 1.25).mean()
        self.delta2 = (maxRatio < 1.25 ** 2).mean()
        self.delta3 = (maxRatio < 1.25 ** 3).mean()

        # self.delta1 = float((maxRatio < 1.25).float().mean())
        # self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        # self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        # self.data_time = 0
        # self.gpu_time = 0
        #
        # inv_output = 1 / output
        # inv_target = 1 / target
        # abs_inv_diff = (inv_output - inv_target).abs()
        # self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        # self.imae = float(abs_inv_diff.mean())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        # self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae, self.sum_rmselg10 = 0, 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        # self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, n=1):
        self.count += n
        # self.sum_irmse += n*result.irmse
        # self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_rmselg10 += n*result.rmselg10
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3

    def average(self):
        avg = Result()
        avg.update(
            self.sum_mae / self.count, self.sum_rmselg10 / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count)

        return avg