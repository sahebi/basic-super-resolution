from __future__ import print_function

import collections
import torch
import os, errno


from .optimizer_lib.lamb.pytorch_lamb import Lamb, log_lamb_rs

CHECKPOINT_DIR = 'checkpoints'
class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()
        self.config = None
        self.CUDA = False
        self.device = 'cpu'
        self.model = None
        self.lr = 0.01
        self.nEpochs = 1
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = 123
        self.upscale_factor = 2
        self.training_loader = None
        self.testing_loader = None

        self.save_path = None
        self.results = collections.defaultdict(list)

    def save_model(self, epoch, avg_error, avg_psnr):
        model_out_path = os.path.join(self.save_path,f"{avg_psnr}_{avg_error}_{epoch}.pth")
        torch.save(self.model, model_out_path)
        with open(os.path.join(self.save_path, '_log.csv'), 'a') as stream:
            stream.writelines(f"{self.config.model}, {self.optimizer.__class__.__name__}, {self.config.nEpochs}, {epoch}, {avg_error}, {avg_psnr}"+os.linesep)
            stream.close()

    def checkpoint_path(self, _type=''):
        optimizer_state_doct = self.optimizer.state_dict()
        if len(optimizer_state_doct) > 0:
            self.save_path = f"{self.config.logprefix}_{self.config.model}_{self.optimizer.__class__.__name__}_{optimizer_state_doct['param_groups'][0]['lr']}"
        else:
            self.save_path = f"{self.config.logprefix}_{self.config.model}_{self.optimizer.__class__.__name__}"

        dest = os.path.join(CHECKPOINT_DIR,self.save_path).lower()

        try:
            os.makedirs(dest)
        except OSError as e:
            pass
        # if not os.path.exists(dest):
        #     os.makedirs(dest)

        with open(os.path.join(dest, '_parameter.txt'), 'w+') as stream:
            stream.write(self.model.__str__())
            stream.write(self.optimizer.__str__())
            stream.close()

        return dest

    def set_optimizer(self, _type='adam'):
        # Set parameters
        _type = self.config.optim

        parameters = {'params': self.model.parameters(),'lr': self.lr}
        # if self.config.momentum > 0 and (_type not in ('sparseadam') or self.config.model not in ('edsr','drcn')):
            # parameters.update({'momentum': self.config.momentum})
        if self.config.weight_decay > 0 and _type not in ('sparseadam','rprop'):
            parameters.update({'weight_decay': self.config.weight_decay})

        # Set optimizar
        if _type == 'adam':
            self.optimizer = torch.optim.Adam(**parameters)
        elif _type == 'sparseadam':
            self.optimizer = torch.optim.SparseAdam(**parameters)
        elif _type == 'adamax':
            self.optimizer = torch.optim.Adamax(**parameters)
        elif _type == 'adam-gamma':
            self.optimizer = torch.optim.Adam(**parameters)
            # parameters.update({'eps': 1e-8})
            # parameters.update({'betas': (0.9, 0.999)})
            # self.optimizer = torch.optim.Adam(**parameters)
        elif _type == 'lamb':
            # parameters.update({'adam': ('adam' == 'adam')})
            parameters.update({'betas': (0.9, 0.999)})
            self.optimizer = Lamb(**parameters)
        elif _type == 'sgd':
            self.optimizer = torch.optim.SGD(**parameters)
        elif _type == 'asgd':
            self.optimizer = torch.optim.ASGD(**parameters)
        elif _type == 'adadelta':
            self.optimizer = torch.optim.Adadelta(**parameters)
        elif _type == 'adagrad':
            self.optimizer = torch.optim.Adagrad(**parameters)
        elif _type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(**parameters)
        elif _type == 'rprop':
            self.optimizer = torch.optim.Rprop(**parameters)

        # if method is GAN, dont call this schedule
        if _type != 'gan':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[500, 750], gamma=0.1)

        self.save_path = self.checkpoint_path(_type=_type)

        return 
