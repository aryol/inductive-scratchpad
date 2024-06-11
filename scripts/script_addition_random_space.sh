#!/bin/bash

mkdir logs-addition-rs
python data/addition/prepare.py


for seed in 1 2 3 4 5 6 7 8 9 10
do
    name=inductive_space_${seed}_10_30
    python train.py config/addition_random_space.py --block_size=1600 --compile=False --seed=$seed --name=$name > logs-addition-rs/${name}.txt
done