#!/bin/bash

# Delta=0-2
python src/run.py -c config/transductive/fb15k237_0-2.yaml --delta 0 --gpus [0]
python src/run.py -c config/transductive/fb15k237_0-2.yaml --delta 1 --gpus [0]
python src/run.py -c config/transductive/fb15k237_0-2.yaml --delta 2 --gpus [0]

# Delta=3-5
python src/run.py -c config/transductive/fb15k237_3-5.yaml --delta 3 --gpus [0]
python src/run.py -c config/transductive/fb15k237_3-5.yaml --delta 4 --gpus [0]
python src/run.py -c config/transductive/fb15k237_3-5.yaml --delta 5 --gpus [0]

# Delta=Specific
python src/run.py -c config/transductive/fb15k237_specific.yaml --delta 2 --gpus [0]