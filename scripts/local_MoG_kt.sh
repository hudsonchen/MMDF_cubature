for seed in 0 1 2 3 4 5 6 7 9
do
  for m in 10
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python kt.py --seed $seed --m $m --dataset mog --step_size 1.0 --bandwidth 1.0
  done
done
