#!/bin/bash

mkdir logs-twocycles-ood
python data/twocycles/prepare.py


for seed in 1 2 3 4 5 6 7 8 9 10
do
    for scratchpad in full induct
    do
        name=ood_${scratchpad}_${seed}
        python train.py config/cycles_ood.py --compile=False --seed=$seed --scratchpad_type=$scratchpad --name=$name > logs-twocycles-ood/${name}.txt
    done
done