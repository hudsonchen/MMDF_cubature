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


# for seed in {21..50}
# do
#   for particle_num in 10 30 100 300 1000
#   do
#     echo "--seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0" >> "$outfile"
#   done
# done

for seed in {0..50}
do
  for particle_num in 10 30 100 300 1000
  do
    if [ "$particle_num" -eq 300 ]; then
    step_num=100000
    elif [ "$particle_num" -eq 1000 ]; then
    step_num=300000
    else
    step_num=10000
    fi
    echo "--seed $seed --particle_num $particle_num --dataset elevators --step_size 1.0 --bandwidth 1.0 --step_num $step_num" >> "$outfile"
  done
done