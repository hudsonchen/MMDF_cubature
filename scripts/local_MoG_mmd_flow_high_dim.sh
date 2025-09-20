for seed in {0..9}
do
  for particle_num in 100
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --d 10 --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 5.0 --inject_noise_scale 0.0 --step_num 100000 --save_path "results_high_dim/"
  done
  for particle_num in 300
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --d 10 --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 5.0 --inject_noise_scale 0.0 --step_num 1000000 --save_path "results_high_dim/"
  done
  for particle_num in 1000
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --d 10 --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 5.0 --inject_noise_scale 0.0 --step_num 3000000 --save_path "results_high_dim/"
  done
done


# for seed in {0..9}
# do
#   for particle_num in 100
#   do
#     /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --d 50 --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 30.0 --inject_noise_scale 0.0 --step_num 300000 --save_path "results_high_dim/"
#   done
#   for particle_num in 300
#   do
#     /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --d 50 --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 30.0 --inject_noise_scale 0.0 --step_num 1000000 --save_path "results_high_dim/"
#   done
#   for particle_num in 1000
#   do
#     /home/zongchen/miniconda3/envs/mmd_cubature/bin/python main.py --seed $seed --d 50 --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 30.0 --inject_noise_scale 0.0 --step_num 3000000 --save_path "results_high_dim/"
#   done
# done

