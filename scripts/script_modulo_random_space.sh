#!/bin/bash

mkdir logs-modulo-length
python data/modulo/prepare.py


for seed in 1 2 3 4 5 6 7 8 9 10
do
    name=inductive_scratchpad_seed_${seed}
    python train.py config/modulo_inductive.py --compile=False --seed=$seed --name=$name > logs-modulo-length/${name}.txt
done