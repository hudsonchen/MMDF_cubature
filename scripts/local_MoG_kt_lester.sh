for seed in 8
do
  for particle_num in 10 30 100 300
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python kt_lester.py --seed $seed --dataset mog --step_size 1.0 --bandwidth 1.0 --particle_num $particle_num
  done
done
