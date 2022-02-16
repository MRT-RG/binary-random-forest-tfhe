#!/bin/bash

for d in {5..20}
do

    echo --- depth ${d} ---
    python3 train.py --depth ${d} --estimators 1 --seed 20220211
    make generate
    make
    make run >> ${OUTPUT_FILE}

done

# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
