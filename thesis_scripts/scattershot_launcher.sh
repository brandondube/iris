#!/bin/bash
END=5*4
for i in $(seq 1 $END); do
    sbatch --export=idx=$i launcher.sh;
done
