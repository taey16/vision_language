
local input_h5 = '/storage/coco/data_342.h5'
local input_json = '/storage/coco/data_342.json'
local total_samples_train = 123287
local total_samples_valid = 3200
local dataset_name = 'coco'

local rnn_size = 384
local num_rnn_layers = 3
local seq_length = -1
local input_encoding_size = 2048
local rnn_type = 'lstm'
local rnn_activation = 'tanh'
local drop_prob_lm = 0.5

local batch_size = 16
local learning_rate = 0.00001--4e-4
local learning_rate_decay_seed = 0.5
local learning_rate_decay_start= 0--300000
local learning_rate_decay_every= 10000--50000

local gpus = {1,2}
local start_from = 
  ''
local experiment_id = string.format(
  'bs%d_%s_%s_hidden%d_layer%d_dropout%.1f_lr%e_anneal_seed%f_start%d_every%d', 
    batch_size, 
    rnn_type, rnn_activation, rnn_size, num_rnn_layers, drop_prob_lm, 
    learning_rate, learning_rate_decay_seed, learning_rate_decay_start, learning_rate_decay_every
)
local checkpoint_path = string.format(
  '/storage/coco/checkpoints/%s_%d_%d_seq_length%d/', 
    dataset_name, total_samples_train, total_samples_valid, seq_length
)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train LM')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_h5',input_h5,
  'path to the h5file containing the preprocessed dataset')
cmd:option('-input_json',input_json,
  'path to the json file containing additional info and vocab')
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
  'rnn type [rnn | lstm | gru | SCRNN(shortcut-RNN)]')
cmd:option('-rnn_activation', rnn_activation,
  'activation for LSTM/RNN [tanh | relu | none]')

-- Optimization: General
cmd:option('-max_iters', -1, 
  'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', batch_size,
  'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip',0.1,
  'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', drop_prob_lm, 
  'strength of dropout in the Language Model RNN')
cmd:option('-seq_per_img',5,
  'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: for the Language Model
cmd:option('-optim','adam',
  'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate', learning_rate,
  'learning rate')
cmd:option('-learning_rate_decay_seed', learning_rate_decay_seed,
  'decay_factor = math.pow(opt.learning_rate_decay_seed, frac)')
cmd:option('-learning_rate_decay_start', learning_rate_decay_start, 
  'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', learning_rate_decay_every, 
  'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,
  'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,
  'beta used for adam')
cmd:option('-optim_epsilon',1e-8,
  'epsilon that goes into denominator for smoothing')


-- Evaluation/Checkpointing
cmd:option('-train_samples', total_samples_train - total_samples_valid, 
  '# of samples in training set')
cmd:option('-val_images_use', total_samples_valid, 
  'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', math.floor((total_samples_train-total_samples_valid)/batch_size/4.0), 
  'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', checkpoint_path, 
  'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 1, 
  'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

-- misc
cmd:option('-gpus', gpus, '# of gpus for cnn')
cmd:option('-id', experiment_id, 
  'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 
  'random number generator seed to use')
cmd:option('-display', 5,
  'display interval for train steps')

cmd:text()

local opt = cmd:parse(arg)

opt.checkpoint_path = paths.concat(opt.checkpoint_path, opt.id)
os.execute('mkdir -p '..opt.checkpoint_path)
print('===> checkpoint path: '..opt.checkpoint_path)

return opt

