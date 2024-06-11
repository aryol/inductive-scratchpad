#!/bin/bash

mkdir logs-twocycles
python data/twocycles/prepare.py


for seed in 1 2 3 4 5
do
    for size in 2 3 4 5 6 7 8 9 10
    do
        for scratchpad in induct full
        do
            name=${scratchpad}_${size}_${seed}
            block=$((size * 16 + 16))
            python train.py config/twocycles.py --compile=False --cycle_size=$size --seed=$seed --block_size=$block --scratchpad_type=$scratchpad --batch_size=512 --learning_rate=0.0003 --name=$name > logs-twocycles/${name}.txt
        done
    done
done