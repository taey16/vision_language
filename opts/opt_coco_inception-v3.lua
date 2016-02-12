
local input_h5 = '/storage/coco/data_342.h5'
local input_json = '/storage/coco/data_342.json'

local use_vgg = false
local torch_model= 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception-v3-2015-12-05/digits_gpu2_inception-v3-2015-12-05_Wed_Jan_27_22_47_34_2016/model_10.bn_removed.t7'
local image_size = 342
local crop_size = 299
local rnn_size = 384
local num_rnn_layers = 3
local input_encoding_size = 2048
local batch_size = 16

local finetune_cnn_after = -1
local learning_rate = 4e-4
local cnn_learning_rate = 1e-5
local cnn_weight_decay = 0.0000001

local start_from = 
  ''
local experiment_id = 
  '_inception-v3-2015-12-05_bn_removed_epoch10_mean_std_modified_bs16_embedding2048_encode384_layer3_lr4e-4'
local gpuid = 0

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5',input_h5,
  'path to the h5file containing the preprocessed dataset')
cmd:option('-input_json',input_json,
  'path to the json file containing additional info and vocab')
cmd:option('-cnn_proto','/storage/models/vgg/vgg_layer16_deploy.prototxt',
  'path to CNN prototxt file in Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-cnn_model','/storage/models/vgg/vgg_layer16.caffemodel',
  'path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-use_vgg', use_vgg, 
  'use vgg16 or not')
cmd:option('-torch_model', torch_model,
  'torch model file path')
cmd:option('-image_size', image_size, 
  'size of input image')
cmd:option('-crop_size', crop_size, 
  'size of croped input image')
cmd:option('-start_from', start_from, 
  'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings
cmd:option('-rnn_size', rnn_size,
  'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',input_encoding_size,
  'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-num_rnn_layers', num_rnn_layers,
  'number of stacks of rnn layers')

-- Optimization: General
cmd:option('-max_iters', -1, 
  'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', batch_size,
  'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,
  'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 
  'strength of dropout in the Language Model RNN')
cmd:option('-finetune_cnn_after', finetune_cnn_after, 
  'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-seq_per_img',5,
  'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: for the Language Model
cmd:option('-optim','adam',
  'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', learning_rate,
  'learning rate')
cmd:option('-learning_rate_decay_start', -1, 
  'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 50000, 
  'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,
  'alpha for adagrad/rmsprop/momentum/adam')
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
cmd:option('-train_samples', 123287 - 3200,
  '# of samples in training set')
cmd:option('-val_images_use', 3200, 
  'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', math.floor(7504/4.0), 
  'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', '/storage/coco/checkpoints', 
  'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 1, 
  'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-losses_log_every', 25, 
  'How often do we snapshot losses (in loss_history), for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', experiment_id, 
  'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 
  'random number generator seed to use')
cmd:option('-gpuid', gpuid, 
  'which gpu to use. -1 = use CPU')
cmd:option('-display', 5,
  'display interval for train steps')

cmd:text()

local opt = cmd:parse(arg)

opt.checkpoint_path = paths.concat(opt.checkpoint_path, opt.id)
os.execute('mkdir -p '..opt.checkpoint_path)
print('===> checkpoint path: '..opt.checkpoint_path)

return opt

