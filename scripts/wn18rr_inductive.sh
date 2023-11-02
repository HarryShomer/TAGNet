#!/bin/bash

# V1-V4
python src/run.py -c config/inductive/wn18rr_v1_best.yaml --version v1 --delta 3 --gpus [0]
python src/run.py -c config/inductive/wn18rr_v2_best.yaml --version v2 --delta 3 --gpus [0]
python src/run.py -c config/inductive/wn18rr_v3_best.yaml --version v3 --delta 3 --gpus [0]
python src/run.py -c config/inductive/wn18rr_v4_best.yaml --version v4 --delta 3 --gpus [0]

# V1-V4 (Delta=Specific)
python src/run.py -c config/inductive/wn18rr_v1_specific_best.yaml --version v1 --delta 3 --gpus [0]
python src/run.py -c config/inductive/wn18rr_v2_specific_best.yaml --version v2 --delta 3 --gpus [0]
python src/run.py -c config/inductive/wn18rr_v3_specific_best.yaml --version v3 --delta 3 --gpus [0]
python src/run.py -c config/inductive/wn18rr_v4_specific_best.yaml --version v4 --delta 3 --gpus [0]
