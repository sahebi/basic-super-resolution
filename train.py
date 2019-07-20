import argparse
import dataset.data as data
from torch.utils.data import DataLoader


from DBPN.solver import DBPNTrainer
from DRCN.solver import DRCNTrainer
from EDSR.solver import EDSRTrainer
from FSRCNN.solver import FSRCNNTrainer
from SRCNN.solver import SRCNNTrainer
from SRCNNT.solver import SRCNNTTrainer
from SRGAN.solver import SRGANTrainer
from SubPixelCNN.solver import SubPixelTrainer
from VDSR.solver import VDSRTrainer
from CARN.solver import CARNTrainer
from MEMNET.solver import MEMNETTrainer
# from result.data import get_training_set, get_test_set

parser = argparse.ArgumentParser(description='Single Image Super Resolution train model')

# Model Parameter
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m',            type=str, default='srcnn', help='choose which model is going to use')
parser.add_argument('--dataset', '-ds',         type=str, default='BSDS300', help='name of dataset defined in dataset.yml')
parser.add_argument('--logprefix', '-l',        type=str, default='', help='name of logfile')
# Train Parameter
parser.add_argument('--batchSize',      type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize',  type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs',        type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr',             type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed',           type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--chanel',         type=int, default=3, help='random seed to use. Default=123')

# Optimization Parameter
parser.add_argument('--optim',          type=str, help='Optimizer Name, sgd, adam, lamb')
parser.add_argument('--momentum',       type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay',   type=float, default=0.01, help='Weight Decay')

# Optimization Scheduler
parser.add_argument('--schduler',       type=str, default='default', help='Scheduler type')
args = parser.parse_args()

def main():
    print('Loading dataset')

    train_set            = data.get_data(dataset_name=args.dataset, data_type='train', upscale_factor=args.upscale_factor)
    test_set             = data.get_data(dataset_name=args.dataset, data_type='test',  upscale_factor=args.upscale_factor)

    # train_set            = data.get_training_set(upscale_factor=args.upscale_factor)
    # test_set             = data.get_test_set(upscale_factor=args.upscale_factor)
    
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader  = DataLoader(dataset=test_set,  batch_size=args.testBatchSize, shuffle=False)

    if args.model == 'sub':
        model = SubPixelTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'srcnn':
        model = SRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'srcnnt':
        model = SRCNNTTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'carn':
        model = SRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'vdsr':
        model = VDSRTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'edsr':
        model = EDSRTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'fsrcnn':
        model = FSRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'drcn':
        model = DRCNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'srgan':
        model = SRGANTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'dbpn':
        model = DBPNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'memnet':
        model = MEMNETTrainer(args, training_data_loader, testing_data_loader)
    else:
        raise Exception("the model does not exist")

    model.run()

if __name__ =='__main__':
    main()


