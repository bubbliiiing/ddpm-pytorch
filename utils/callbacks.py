import os

import matplotlib
import torch

matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        # try:
        #     for m in model:
        #         dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
        #         self.writer.add_graph(m, dummy_input)
        # except:
        #     pass

    def append_loss(self, epoch, **kwargs):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, [])
            #---------------------------------#
            #   为列表添加数值
            #---------------------------------#
            getattr(self, key).append(value)
        
            #---------------------------------#
            #   写入txt
            #---------------------------------#
            with open(os.path.join(self.log_dir, key + ".txt"), 'a') as f:
                f.write(str(value))
                f.write("\n")
                
            #---------------------------------#
            #   写入tensorboard
            #---------------------------------#
            self.writer.add_scalar(key, value, epoch)
            
        self.loss_plot(**kwargs)

    def loss_plot(self, **kwargs):
        plt.figure()
        
        for key, value in kwargs.items():
            losses = getattr(self, key)
            plt.plot(range(len(losses)), losses, linewidth = 2, label = key)

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")