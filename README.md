# Dataset preperation
1. Run `./data/attribute/prepro_attribute.py`
	* **check argument** `--input_filename # output file from shuffle_duplicate_remove.py`
	* **check argument** `--output_json, # resulting output_json file. this file finally used for train/val/test`
	* **check argument** `--output_h5 # resulting output_h5 file. this file also used for train/val/test`
	* check argument `image_dim, # default: 342` 

# Training/val
1. Move to the directory, `cd /path/to/the/git/repo.`
2. Change options, `vim opts/opt_attribute.lua`
	* **check argument** `input_h5 --output h5 file of prepro_attribute.py`
	* **check argument** `input_json --output json file of prepro_attribute.py`
	* **check argument** `torch_model --path to pretrained-image encoder network`
	* check argument `gpus = {1,2,3,4}`
	* check argument for image-encoder `image_size, crop_size, crop_jitter, flip_jitter = 342, 299, true, false`
	* tune hyperparameters for the language-decoder (e.g. `rnn_size, num_rnn_layers and drop_prob_lm use_bn='original'` etc)
	* tune hyperparameters for optimisation (e.g. `batch_size, optimizer, learning_rate` etc)

# Launching training/val scripts
1. Suppose you want to run on 4 gpu cards
	* **check argument** `local gpus = {1,2,3,4} --in opts/opt_xxx.lua`
	* `CUDA_VISIBLE_DEVICES=0,1,2,3 luajit train.lua`

# Plotting learning curve
1. Both log-files for train/val are saved in your checkpoint path automatically. 
