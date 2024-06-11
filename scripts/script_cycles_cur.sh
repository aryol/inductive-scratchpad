#!/bin/bash

mkdir logs-twocycles-cur
python data/twocycles/prepare.py


for seed in 1 2 3 4 5
do
    for mode in multiple_nodes multiple_nodes_cur multiple_nodes_cur_forget # Mixed distribution, curriculum on mixed distribution without forgetting, curriculum on mixed distribution with forgetting
    do
        name=${mode}_${seed}
        python train.py config/cycles_cur.py --compile=False --seed=$seed --mode=$mode --name=$name > logs-twocycles-cur/${name}.txt
    done
done
