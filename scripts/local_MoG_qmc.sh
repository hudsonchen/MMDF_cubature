for seed in {0..1}
do
  for particle_num in 10 30 100 300 1000
  do
    /home/zongchen/miniconda3/envs/mmd_cubature/bin/python qmc.py --seed $seed --particle_num $particle_num --dataset mog
  done
done

