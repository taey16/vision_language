
local input_h5 = 
  '/storage/freebee/tshirts_shirts_blous_knit.image_sentence.txt.h5'
  --'/storage/freebee/tshirts_shirts_blous.image_sentence.txt.h5'
  --'/storage/freebee/tshirts_shirts.image_sentence.txt.h5'
  --'/storage/freebee/tshirts_excel_1453264869210.csv.image_sentence.txt.h5'
local input_json = 
  '/storage/freebee/tshirts_shirts_blous_knit.image_sentence.txt.json'
  --'/storage/freebee/tshirts_shirts_blous.image_sentence.txt.json'
  --'/storage/freebee/tshirts_shirts.image_sentence.txt.json'
  --'/storage/freebee/tshirts_excel_1453264869210.csv.image_sentence.txt.json'
local total_samples_train = 103607
local total_samples_valid = 8000
local dataset_name = 'tshirts_shirts_blous_knit'

local torch_model= 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/model_33.bn_removed.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/model_31.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Sat_Jan_30_17_16_06_2016/model_16.bn_removed.t7'
local image_size = 342
local crop_size = 299
local flip_jitter = 0

local rnn_size = 256
local num_rnn_layers = 2
local seq_length = 2
local input_encoding_size = 2048
local rnn_activation = 'relu'
local rnn_type = 'rnn'
local drop_prob_lm = 0.5

local batch_size = 16
local finetune_cnn_after = -1
local learning_rate = 4e-4
local learning_rate_decay_start = 100000
local cnn_learning_rate = 1e-5
local cnn_weight_decay = 0.0000001

local start_from = 
  ''
local experiment_id = string.format(
  '_inception-v3-2015-12-05_bn_removed_epoch33_bs%d_flip%d_%s_%s_hidden%d_layer%d_dropout%.1f_lr%e_anneal_%d', batch_size, flip_jitter, rnn_type, rnn_activation, rnn_size, num_rnn_layers, drop_prob_lm, learning_rate, learning_rate_decay_start
)
local checkpoint_path = string.format(
  '/storage/attribute/checkpoints/%s_%d_%d_seq_length%d/', dataset_name, total_samples_train, total_samples_valid, seq_length
)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5',input_h5,
  'path to the h5file containing the preprocessed dataset')
cmd:option('-input_json',input_json,
  'path to the json file containing additional info and vocab')
cmd:option('-torch_model', torch_model,
  'torch model file path')
cmd:option('-image_size', image_size, 
  'size of input image')
cmd:option('-crop_size', crop_size, 
  'size of croped input image')
cmd:option('-flip_jitter', flip_jitter, 
  'flag for flipping [0 | 1 ]')
cmd:option('-start_from', start_from, 
  'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size', rnn_size,
  'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',input_encoding_size,
  'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-num_rnn_layers', num_rnn_layers,
  'number of stacks of rnn layers')
cmd:option('-seq_length', seq_length,
  'number of seq. length (without EOS/SOS token)')
cmd:option('-rnn_type', rnn_type,
  'rnn type [rnn | lstm | gru]')
cmd:option('-rnn_activation', rnn_activation,
  'activation for LSTM/RNN [tanh | relu | none]')

-- Optimization: General
cmd:option('-max_iters', -1, 
  'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', batch_size,
  'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip', 0.1,
  'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', drop_prob_lm, 
  'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', finetune_cnn_after, 
  'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img', 1,
  'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: for the Language Model
cmd:option('-optim','adam',
  'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', learning_rate,
  'learning rate')
cmd:option('-learning_rate_decay_start', learning_rate_decay_start, 
  'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 
  'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,
  'alpha for adagrad/rmsprop/momentum/adam (i.e. stepsize, learningrate')
cmd:option('-optim_beta',0.999,
  'beta used for adam')
cmd:option('-optim_epsilon',1e-8,
  'epsilon that goes into denominator for smoothing')

-- Optimization: for the CNN
cmd:option('-cnn_optim','adam',
  'optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,
  'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,
  'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate', cnn_learning_rate,
  'learning rate for the CNN')
cmd:option('-cnn_weight_decay', cnn_weight_decay, 
  'L2 weight decay just for the CNN')

-- Evaluation/Checkpointing
cmd:option('-train_samples', total_samples_train - total_samples_valid,
  '# of samples in training set')
cmd:option('-val_images_use',total_samples_valid,
  'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', math.floor((total_samples_train - total_samples_valid) / batch_size /2.0), 
  'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', checkpoint_path, 
  'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 0, 
  'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 0, 
  'How often do we snapshot losses (in loss_history), for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', experiment_id, 
  'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 
  'random number generator seed to use')
cmd:option('-gpuid', 0, 
  'which gpu to use. -1 = use CPU')
cmd:option('-display', 5,
  'display interval for train steps')

cmd:text()

local opt = cmd:parse(arg)

opt.checkpoint_path = paths.concat(opt.checkpoint_path, opt.id)
os.execute('mkdir -p '..opt.checkpoint_path)
print('===> checkpoint path: '..opt.checkpoint_path)

return opt

