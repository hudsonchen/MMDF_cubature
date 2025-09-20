for seed in {0..9}
do
  for particle_num in 10 30 100 300 1000
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python kh.py --seed $seed --particle_num $particle_num --dataset elevators --step_size 1.0 --bandwidth 1.0 --kernel Matern_32
  done
done


for seed in {0..9}
do
  for particle_num in 10 30 100 300 1000
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python kh.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --kernel Matern_32
  done
done