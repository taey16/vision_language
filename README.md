# Launching training/val scripts
1. Suppose you want to run on 4 gpu cards
	* `CUDA_VISIBLE_DEVICES=0,1,2,3 luajit train.lua`

# Plotting learning curve
1. Both train and val. log files are saved in your checkpoint path automatically 
