from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from DBPN.model import DBPN, DBPNS, DBPNLL
from progress_bar import progress_bar

from Trainer import Trainer

class DBPNTrainer(Trainer):
    def __init__(self, config, training_loader, testing_loader):
        super(DBPNTrainer, self).__init__()
        self.config = config
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.model = DBPN(num_channels=1, base_channels=64, feat_channels=256, num_stages=7,
                            scale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init()
        self.criterion = nn.L1Loss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.set_optimizer()

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            total_time = progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        avg_loss = train_loss / len(self.training_loader)
        return [avg_loss, total_time]

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                total_time = progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        avg_psnr = avg_psnr / len(self.testing_loader)
        return [avg_psnr, total_time]

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            avg_loss = self.train()
            avg_psnr = self.test()
            self.scheduler.step(epoch)
            self.save_model(epoch=epoch, avg_error=avg_loss, avg_psnr=avg_psnr)
