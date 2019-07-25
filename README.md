### Basic Super Resolution

This git cloned from https://github.com/icpm/super-resolution change some modifications.

- [x] Add CARN method
- [x] Add different optimization method
- [x] Log the checkppoints and _logs
- [ ] Log the result

### Optimizer
- ADAM
- AdamSparse
- Adamax
- Adadelta
- Adagrad
- ASGD
- LAMB
- RProp
- SGD
- RMSprop

### Single Image Super Resolution Methods
- [X] SubPixelCNN
- [X] SRCNN
- [X] SRCNNT
- [X] VDSR
- [X] EDSR
- [X] FSRCNN
- [X] DRCN `batchsize should be small, batchsize=4`
- [ ] SRGAN `change save checkpoint path patterns`
- [X] DBPN `batchsize should be small, batchsize=1`
- [X] MemNet
- [ ] CARN

##### Mount google drive install basic super resolution package
```
import os
from google.colab import drive
drive.mount('/content/gdrive/')

!pip install tensorboardX

os.chdir('/content/gdrive/My Drive/Projects/master-code')
!git clone https://github.com/sahebi/basic-super-resolution
```

##### Run train all python script
```
os.chdir('/content/gdrive/My Drive/Projects/master-code/basic-super-resolution')

!python train_all.py --logprefix test_BSDS300_2x -uf 2 --dataset BSDS300 --batchSize 128 --testBatchSize 128 --nEpochs 1500 --iter 3
```

#### Train Model
`python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model srcnnt`

`python train.py --logprefix test1epoch -uf 4 --dataset COCO --batchSize 16 --testBatchSize 8 --nEpochs 1 --model srcnnt`

### Run super resolution
`python super_resolve.py --input result/BSD300_3096.jpg/4x/lr.jpg --model model/carn_488.pth --output result/BSD300_3096.jpg/4x/carn.jpg`

