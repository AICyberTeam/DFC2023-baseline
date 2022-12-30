import torch
import math
import numpy as np

def log10(x):
      """Convert a new tensor with the base-10 logarithm of the elements of x. """
      return torch.log(x+1e-6) / math.log(10)

class Result(object):
    def __init__(self):
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def set_to_worst(self):
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def update(self, mse, rmse, mae, delta1, delta2, delta3):
        self.mse, self.rmse, self.mae= mse, rmse, mae
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3


    def evaluate(self, output, target):
        output[output == 0] = 0.00001
        output[output < 0] = 999
        target[target <= 0] = 0.00001
        valid_mask = ((target>0) + (output>0)) > 0

        output = output[valid_mask]
        target = target[valid_mask]
        abs_diff = np.abs(output - target)

        self.mse = np.mean(abs_diff ** 2)
        self.rmse = np.sqrt((abs_diff ** 2).mean())
        self.mae = np.mean(abs_diff)

        maxRatio = np.maximum(output / target, target / output)
        self.delta1 = (maxRatio < 1.25).mean()
        self.delta2 = (maxRatio < 1.25 ** 2).mean()
        self.delta3 = (maxRatio < 1.25 ** 3).mean()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0

    def update(self, result, n=1):
        self.count += n
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3

    def average(self):
        avg = Result()
        avg.update(
            self.sum_mae / self.count, self.sum_mse / self.count, self.sum_rmse / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count)

        return avg
