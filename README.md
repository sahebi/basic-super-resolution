### Basic Super Resolution

This git cloned from https://github.com/icpm/super-resolution change some modifications.

[x] Add CARN method
[x] Add different optimization method
[x] Log the result
[ ] 

### Optimizer
- LAMB
- SGD
- ADAM
- AdamSparse
- Adamax
- asgd
- Adadelta
- adagrad
- rmsprop
- rprop

### Single Image Super Resolution Methods
- SubPixelCNN
- SRCNN
- SRCNNT
- CARN
- VDSR
- EDST
- FSRCNN
- DRCN
- SRGAN
- DBPN
- MemNet

##### Mount google drive install basic super resolution package
```python
import os
from google.colab import drive
drive.mount('/content/gdrive/')

!pip install tensorboardX

os.chdir('/content/gdrive/My Drive/Projects/master-code')
!git clone https://github.com/sahebi/basic-super-resolution
```

##### Run train all python script
```python
os.chdir('/content/gdrive/My Drive/Projects/master-code/basic-super-resolution')
!python train_all.py --logprefix test_BSDS300_4x -uf 4 --dataset BSDS300 --batchSize 64 --testBatchSize 64 --nEpochs 1000 --iter 3
```

#### Train Model
`python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model srgan`

`python train.py --logprefix test1epoch -uf 4 --dataset COCO --batchSize 16 --testBatchSize 8 --nEpochs 1 --model srcnnt`

### Run super resolution
`python super_resolve.py --input result/BSD300_3096.jpg/4x/lr.jpg --model model/carn_488.pth --output result/BSD300_3096.jpg/4x/carn.jpg`

