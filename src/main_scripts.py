import os

# os.system('CUDA_VISIBLE_DEVICES=2 python3 main.py '
#           '--mode flatcv --data ADNI '
#           '--lr 0.05 --reassign 1 --epochs 100')

os.system('CUDA_VISIBLE_DEVICES=2 python3 main.py '
          '--mode test --model final/checkpoint_epoch0999.pth.tar '
          '--gmmmodel final/gm_params.mat')
