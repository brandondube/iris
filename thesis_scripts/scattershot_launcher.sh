#!/bin/bash
END=5*4;
for ((i=1; i<=$END; i++)); do
    sbatch --export=idx=$i launcher.sh;
done
