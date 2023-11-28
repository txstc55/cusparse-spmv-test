#!/bin/bash

matrixSize=(5000 10000 20000 40000 80000 160000 320000 640000 1280000 2560000 5120000)

# matrixSize=(500 1000 2000 4000)


rm -rf performance.txt
for matSize in ${matrixSize[@]}
do
    python generateSparseIndices.py $matSize 12 3
    ./run.sh >> performance.txt
done