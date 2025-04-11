# rsync -ruP hudson_ucl:/home/zongchen/mmd_flow_cubature/figures/ /Users/hudsonchen/Desktop/figures/

for method in "kernel_herding" "kernel_thinning"; do
    rsync -ruP myriad:/home/ucabzc9/Scratch/mmd_flow_cubature/results/${method}/house_8L_dataset/Gaussian_kernel/ /home/zongchen/mmd_flow_cubature/results/${method}/house_8L_dataset/Gaussian_kernel/
done

