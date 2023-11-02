#!/bin

# Assumes in scripts/ folder
cd ../astarnet

# FB15k-237
python script/run.py -c config/transductive/fb15k237_astarnet.yaml --gpus [0]

# WN18RR
python script/run.py -c config/transductive/wn18rr_astarnet.yaml --gpus [0]