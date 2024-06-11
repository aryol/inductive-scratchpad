#!/bin/bash

mkdir logs-modulo-sp
python data/modulo/prepare.py

find_closest_multiple() {
    local n=$1
    local closest_multiple=$(( (n + 31) / 32 * 32 ))
    echo $closest_multiple
}

for seed in 1 2 3 4 5 6 7 8 9 10
do
    for n in 40 80 120 160 200
    do
        degree=$((n))
        dim=$((2*n))
        embedding_dim=$((2*n))
        name=p_${seed}_${n}_1024
        block_size=$(find_closest_multiple $((3*n+8)))
        python train.py config/modulo.py --compile=False --name=${name} --block_size=$block_size --degree=$degree --dim=$dim --embedding_dim=$embedding_dim --seed=$seed > logs-modulo-sp/${name}.txt
    done
done
