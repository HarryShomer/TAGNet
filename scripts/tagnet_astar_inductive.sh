#!/bin

# Assumes in scripts/ folder
cd ../astarnet

# FB15k-237 versions v1-v4
python script/run.py -c config/inductive/fb15k237_astarnet.yaml --version v1 --gpus [0]
python script/run.py -c config/inductive/fb15k237_astarnet.yaml --version v2 --gpus [0]
python script/run.py -c config/inductive/fb15k237_astarnet.yaml --version v3 --gpus [0]
python script/run.py -c config/inductive/fb15k237_astarnet.yaml --version v4 --gpus [0]

# WN18RR (6, 7, 8 layers)
python script/run.py -c config/inductive/wn18rr_astarnet.yaml --version v1 --gpus [0]
python script/run.py -c config/inductive/wn18rr_astarnet.yaml --version v2 --gpus [0]
python script/run.py -c config/inductive/wn18rr_astarnet.yaml --version v3 --gpus [0]
python script/run.py -c config/inductive/wn18rr_astarnet.yaml --version v4 --gpus [0]