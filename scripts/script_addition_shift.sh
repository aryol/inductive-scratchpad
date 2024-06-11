#!/bin/bash

mkdir logs-addition-shift
python data/addition/prepare.py


for seed in 1 2 3 4 5 6 7 8 9 10
do
    name=inductive_shift_${seed}_4_30
    python train.py config/addition_shift.py --compile=False --seed=$seed --name=$name > logs-addition-shift/${name}.txt
done