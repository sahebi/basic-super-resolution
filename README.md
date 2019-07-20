*Train Model*
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model srcnnt
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model sub
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 32 --testBatchSize 8 --nEpochs 500 --model carn
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model srcnn
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model vdsr
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model edsr
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model fsrcnn
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 16 --testBatchSize 8 --nEpochs 1 --model srgan

python train.py --logprefix test1epoch -uf 4 --dataset COCO --batchSize 16 --testBatchSize 8 --nEpochs 1 --model srcnnt

`Very High GPU memory needed`
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 1 --testBatchSize 8 --nEpochs 1 --model dbpn
python train.py --logprefix testmodel -uf 4 --dataset BSDS300 --batchSize 1 --testBatchSize 8 --nEpochs 1 --model drcn

### Run super resolution
`
python super_resolve.py --input result/BSD300_3096.jpg/4x/lr.jpg --model model/carn_488.pth --output result/BSD300_3096.jpg/4x/carn.jpg
`

### Optimizer
- SGD
- ADAM
- LAMB
- AdamSparse

### Read About BERT
python train_all.py --logprefix test_BSDS300_4x -uf 4 --dataset BSDS300 --batchSize 8 --testBatchSize 8 --nEpochs 500

python train_all.py --logprefix test_bs8_2x_BSDS300 -uf 2 --dataset BSDS300 --batchSize 8 --testBatchSize 8 --nEpochs 500