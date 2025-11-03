for seed in 10
do
  for m in 10
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python kt.py --seed $seed --m $m --dataset house_8L --step_size 1.0 --bandwidth 1.0 --kernel Gaussian
  done
done

# for seed in {11..50}
# do
#   for m in 10
#   do
#     /home/zongchen/miniconda3/envs/mmd_cubature/bin/python kt.py --seed $seed --m $m --dataset elevators --step_size 1.0 --bandwidth 1.0 --kernel Gaussian
#   done
# done