# rsync -ruP hudson_ucl:/home/zongchen/mmd_flow_cubature/figures/ /Users/hudsonchen/Desktop/figures/

for method in "kernel_herding" "kernel_thinning" "support_points"; do
    rsync -ruP myriad:/home/ucabzc9/Scratch/mmd_flow_cubature/results/${method}/elevators_dataset/Gaussian_kernel/ /home/zongchen/mmd_flow_cubature/results/${method}/elevators_dataset/Gaussian_kernel/
done

for method in "kernel_herding" "kernel_thinning" "support_points"; do
    rsync -ruP myriad:/home/ucabzc9/Scratch/mmd_flow_cubature/results/${method}/house_8L_dataset/Gaussian_kernel/ /home/zongchen/mmd_flow_cubature/results/${method}/house_8L_dataset/Gaussian_kernel/
done