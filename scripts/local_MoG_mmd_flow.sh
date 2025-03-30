# for seed in {0..9}
# do
#   for particle_num in 100
#   do
#     /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.0 --step_num 100000
#   done
#   for particle_num in 300
#   do
#     /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.0 --step_num 1000000
#   done
#   for particle_num in 1000
#   do
#     /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0 --inject_noise_scale 0.0 --step_num 3000000
#   done
# done

# This is the code for exactness figure
for seed in {1..9}
do
  for particle_num in 100
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --particle_num $particle_num --dataset mog --step_size 2.0 --bandwidth 1.0 --inject_noise_scale 0.0 --step_num 30000000
  done
done

