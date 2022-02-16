#!/bin/bash

# Run experiments with different hyper parameters.

for d in {3..8}
do

    echo --- depth ${d} ---
    python3 train.py --depth ${d} --seed 20220211
    make

done

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
