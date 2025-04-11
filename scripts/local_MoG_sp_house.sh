for seed in {11..20}
do
  for particle_num in 10 30 100
    do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python support_points.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --step_num 10000
    done
  for particle_num in 300
    do
        /home/zongchen/miniconda3/envs/mmd_cubature/bin/python support_points.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --step_num 100000
    done
  for particle_num in 1000
    do
        /home/zongchen/miniconda3/envs/mmd_cubature/bin/python support_points.py --seed $seed --particle_num $particle_num --dataset house_8L --step_size 1.0 --bandwidth 1.0 --step_num 300000
    done
done
