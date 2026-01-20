#!/bin/bash
# run_tc_weights.sh
# Lancia fldd.py con diversi valori di tc_weight in background
# Log in logs/tc_<value>.log

# Lista dei valori di tc_weight
tc_weights=(0.1 0.0001)

# Crea cartella logs se non esiste
mkdir -p logs

# Loop sui valori e lancio job in background
for tc in "${tc_weights[@]}"; do
    echo "Launching job for tc_weight=$tc ..."
    srun -A deep_learning -t 120 \
        python fldd.py --tc_weight $tc \
        > logs/tc_${tc}.log 2>&1 &
done

echo "All jobs launched!"
