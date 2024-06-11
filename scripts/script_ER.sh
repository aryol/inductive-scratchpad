#!/bin/bash

mkdir logs-ER
python data/twocycles/prepare.py


for seed in 1 2 3 4 5 6 7 8 9 10
do
    name=ER_${seed}
    python train.py config/ER.py --compile=False --seed=$seed --name=$name > logs-ER/${name}.txt
done