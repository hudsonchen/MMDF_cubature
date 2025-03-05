# for seed in {0..10}
# do
#   for particle_num in 10 30 100 300 1000
#   do
#     /home/zongchen/miniconda3/envs/mmd_cubature/bin/python dpp.py --seed $seed --particle_num $particle_num --dataset mog --step_size 1.0 --bandwidth 1.0
#   done
# done

for seed in {0..10}
do
  for particle_num in 10 30 100 300 1000
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python dpp.py --seed $seed --particle_num $particle_num --dataset gaussian --step_size 1.0 --bandwidth 1.0
  done
done