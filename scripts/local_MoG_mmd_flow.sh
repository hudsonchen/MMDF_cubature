for seed in {0..10}
do
  for particle_num in 10 30 100
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.1 --step_num 10000
  done
  for particle_num in 300 1000
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.1 --step_num 30000
  done
done
