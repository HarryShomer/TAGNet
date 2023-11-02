#!/bin/bash

# Delta=0-5
python src/run.py -c config/transductive/wn18rr.yaml --delta 0 --gpus [0]
python src/run.py -c config/transductive/wn18rr.yaml --delta 1 --gpus [0]
python src/run.py -c config/transductive/wn18rr.yaml --delta 2 --gpus [0]
python src/run.py -c config/transductive/wn18rr.yaml --delta 3 --gpus [0]
python src/run.py -c config/transductive/wn18rr.yaml --delta 4 --gpus [0]
python src/run.py -c config/transductive/wn18rr.yaml --delta 5 --gpus [0]

# Delta=Specific
python src/run.py -c config/transductive/wn18rr_specific.yaml --delta 3 --gpus [0]