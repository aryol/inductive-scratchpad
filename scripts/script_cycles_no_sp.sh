#!/bin/bash


mkdir logs-twocycles
python data/twocycles/prepare.py

for seed in 1 2 3 4 5
do
    for size in 2 3 4 5 6 7
    do
        name=mini_${size}_no_sp_${seed}
        python train.py config/twocycles_xs_no_sp.py --compile=False --cycle_size=$size --seed=$seed --name=$name > logs-twocycles/${name}.txt
        name=small_${size}_no_sp_${seed}
        python train.py config/twocycles_s_no_sp.py --compile=False --cycle_size=$size --seed=$seed --name=$name > logs-twocycles/${name}.txt
        name=medium_${size}_no_sp_${seed}
        python train.py config/twocycles_m_no_sp.py --compile=False --cycle_size=$size --seed=$seed --name=$name > logs-twocycles/${name}.txt
    done
done