#!/bin/bash

outfile=$1  # output filename e.g., job_list.txt

> "$outfile"  # empty the file first

# for seed in {21..50}
# do
#   for m in 4 5 6 7 8 9
#   do
#     echo "--seed $seed --m $m --dataset house_8L --step_size 1.0 --bandwidth 1.0" >> "$outfile"
#   done
# done


for seed in {21..50}
do
  for particle_num in 10 30 100 300 1000
  do
    echo "--seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0" >> "$outfile"
  done
done
