require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
local path = require 'pl.path'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local parallel_utils = require 'misc.parallel_utils'
require 'misc.DataLoader'
require 'misc.optim_updates'
require 'models.LanguageModel'
require 'models.FeatExpander'
require 'optim'

local opt = paths.dofile('opts/opt_coco_inception-v3-default.lua')
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.manualSeedAll(opt.seed)

local loader = DataLoader{h5_file = opt.input_h5, 
                          json_file = opt.input_json}

local protos = {}
if string.len(opt.start_from) > 0 then
  -- load protos from file
  io.flush(print('initializing weights from ' .. opt.start_from))
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do 
    net_utils.unsanitize_gradients(v) 
  end
  -- not in checkpoints, create manually
  protos.crit = nn.LanguageModelCriterion()
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  cudnn.convert(protos.cnn, cudnn)
else
  -- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.use_bn = opt.use_bn
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = opt.num_rnn_layers
  lmOpt.dropout = opt.drop_prob_lm
  data_seq_length = loader:getSeqLength()
  if data_seq_length ~= opt.seq_length then
    io.flush(print(string.format(
      'data_seq_length: %d, opt.seq_length: %d', data_seq_length, opt.seq_length)))
    if opt.seq_length == -1 then
      io.flush(print(string.format(
        'we will use opt.seq_length: %d as dataloader said', data_seq_length)))
      opt.seq_length = data_seq_length
    end
  end
  lmOpt.seq_length = opt.seq_length
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  lmOpt.rnn_activation = opt.rnn_activation
  lmOpt.rnn_type = opt.rnn_type
  protos.lm = nn.LanguageModel(lmOpt)
  if opt.embedding_model and opt.embedding_model ~= '' then
    local embedding_checkpoint = torch.load(opt.embedding_model)
    local protos_embedding = embedding_checkpoint.protos.lm
    if protos.lm.lookup_table.weight then
      io.flush(print('initializing embedding weights from '.. opt.embedding_model))
      protos.lm.lookup_table.weight:copy(protos_embedding.lookup_table.weight)
    end
    if protos.lm.lookup_table.bias then
      io.flush(print('initializing embedding bias from '.. opt.embedding_model))
      protos.lm.lookup_table.bias:copy(protos_embedding.lookup_table.bias)
    end
  end
  -- initialize the pretrained ConvNet
  protos.cnn = net_utils.build_cnn({encoding_size = opt.input_encoding_size, 
                                    model_filename = opt.torch_model}
  )
  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()
end

cudnn.benchmark = true
cudnn.fastest = true
if #opt.gpus > 1 then
  protos.cnn = parallel_utils.makeDataParallel(protos.cnn, opt.gpus)
end
-- ship everything to GPU, maybe
for k,v in pairs(protos) do v:cuda() end
print(protos.cnn)

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(params:nElement() == grad_params:nElement(), 
  'ERROR check params:nElement() == grad_params:nElement()')
assert(cnn_params:nElement() == cnn_grad_params:nElement(),
  'ERROR check cnn_params:nElement() == cnn_grad_params:nElement()')

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
thin_lm.core:share(protos.lm.core, 'weight', 'bias', 'running_mean', 'running_std', 'running_var')
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn
if #opt.gpus > 1 then
  thin_cnn = protos.cnn:get(1):clone()
  thin_cnn:share(protos.cnn:get(1), 'weight', 'bias', 'running_mean', 'running_std', 'running_var')
else
  thin_cnn = protos.cnn:clone()
  thin_cnn:share(protos.cnn, 'weight', 'bias', 'running_mean', 'running_std', 'running_var')
end
-- sanitize all modules of gradient storage so that we dont save big checkpoints
net_utils.sanitize_gradients(thin_cnn)
local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this 
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around.
protos.lm:createClones()
collectgarbage()


-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)
  local bn_rnn = utils.getopt(evalopt, 'use_bn')
  local opt_iter = utils.getopt(evalopt, 'iter')
  local checkpoint_path = utils.getopt(evalopt, 'checkpoint_path')

  protos.cnn:evaluate()
  -- it doesnt seemd to work when using BN-LSTM
  if bn_rnn == 'original' then
    protos.lm:evaluate()
  end

  -- rewind iteator back to first datapoint in the split
  loader:resetIterator(split)
  local n = 0
  local loss_sum = 0
  local logprobs_sum = 0
  local perplexity = 0
  local precision = 0
  local loss_evals = 0
  local predictions = {}
  local vocab = loader:getVocab()

  while true do
    -- fetch a batch of data
    local data = loader:getBatch{
      batch_size = opt.batch_size, 
      image_size = opt.image_size, 
      split = split, 
      seq_per_img = opt.seq_per_img
    }

    data.labels = data.labels[{{1,opt.seq_length},{}}]

    -- preprocess in place, and don't augment
    data.images = net_utils.preprocess(data.images, 
                                       opt.crop_size, 
                                       false, 
                                       false)
    n = n + data.images:size(1)

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    local expanded_feats = protos.expander:forward(feats)
    local logprobs = protos.lm:forward{expanded_feats, data.labels}
    local loss = protos.crit:forward(logprobs, data.labels)
    local acc, pplx = 0, 0
    acc, pplx = protos.crit:precision(logprobs, data.labels)
    loss_sum = loss_sum + loss
    --logprobs_sum = logprobs_sum + logprobs
    loss_evals = loss_evals + 1
    precision = precision + acc[2]
    perplexity = perplexity + pplx

    -- forward the model to also get generated samples for each image
    local seq = protos.lm:sample(feats)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents/16 do
      local entry = {image_id = data.infos[k].id, 
                     file_path = data.infos[k].file_path, 
                     caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        io.flush(print(string.format('image %s(%s): %s', 
                                     entry.image_id, 
                                     entry.file_path, 
                                     entry.caption)))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      io.flush(print(string.format(
        'evaluating validation performance... %d/%d (loss: %f, acc[2]: %f, pplx: %f)', 
        ix0-1, ix1, loss, acc[2], pplx)))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end
  if opt.tsne == 1 then
    net_utils.tsne_embedding(thin_lm.lookup_table, 
                             vocab, 
                             opt_iter, 
                             checkpoint_path)
  end

  return loss_sum/loss_evals, 
         predictions, 
         lang_stats, 
         perplexity/loss_evals, 
         precision/loss_evals
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun(finetune_cnn)
  protos.cnn:training()
  protos.lm:training()
  grad_params:zero()
  if finetune_cnn then
    cnn_grad_params:zero()
  end

  -----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{
    batch_size = opt.batch_size, 
    image_size = opt.image_size, 
    split = 'train', 
    seq_per_img = opt.seq_per_img
  }

  data.labels = data.labels[{{1,opt.seq_length},{}}]

  -- preproces in-place, data augment in training
  data.images = net_utils.preprocess(data.images, 
                                     opt.crop_size, 
                                     opt.crop_jitter, 
                                     opt.flip_jitter)

  -- data.images: Nx3xopt.image_sizexopt.image_size
  -- data.seq: LxM where L is sequence length upper bound, and M = N*seq_per_img

  -- forward the ConvNet on images (most work happens here)
  local feats = protos.cnn:forward(data.images)
  -- we have to expand out image features, once for each sentence
  local expanded_feats = protos.expander:forward(feats)
  -- forward the language model
  local logprobs = protos.lm:forward{expanded_feats, data.labels}
  -- forward the language model criterion
  local loss = protos.crit:forward(logprobs, data.labels)
  -- compute perplexity
  local perplexity, precision = 0, 0
  precision, perplexity = protos.crit:precision(logprobs, data.labels)
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  -- backprop language model
  local dexpanded_feats, ddummy = 
    unpack(protos.lm:backward({expanded_feats, data.labels}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  if finetune_cnn then
    local dfeats = protos.expander:backward(feats, dexpanded_feats)
    local dx = protos.cnn:backward(data.images, dfeats)
  end

  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -- apply L2 regularization
  if finetune_cnn and opt.cnn_weight_decay > 0 then
    cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    -- note: we don't bother adding the l2 loss to the total loss, meh.
    cnn_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end
  -----------------------------------------------------------------------------

  return {total_loss = loss, 
          total_perplexity = perplexity, 
          precision = precision}
end

local logger_trn = 
  optim.Logger(paths.concat(opt.checkpoint_path, 'train.log'))
local logger_tst = 
  optim.Logger(paths.concat(opt.checkpoint_path, 'test.log'))


-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local iter = opt.retrain_iter
local loss0
local number_of_batches = opt.train_samples / opt.batch_size
local optim_state = {}
local cnn_optim_state = {}
local best_score
local tm = torch.Timer()

while true do  
  local start_trn = tm:time().real

  local finetune_cnn = false
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    finetune_cnn = true
  end

  -- eval loss/gradient
  local losses = lossFun(finetune_cnn)

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and 
     opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(opt.learning_rate_decay_seed, frac)
    learning_rate = learning_rate * decay_factor
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  if opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'nag' then
    nag(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  else
    error('bad option opt.optim', opt.optim)
  end

  if finetune_cnn then
    if opt.cnn_optim == 'nag' then
      nag(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'adam' then
      adam( cnn_params, cnn_grad_params, 
        cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim', opt.cnn_optim)
    end
  end

  local elapsed_trn = tm:time().real - start_trn
  local epoch = iter * 1.0 / number_of_batches
  if iter % opt.display == 0 then
    --local embedding_weight = protos.lm.lookup_table.weight:float():clone()
    --print(embedding_weight[{{10},{1,5}}])
    --print(embedding_weight[{{20},{1,5}}])
    io.flush(print(string.format(
      '%d/%d: %.2f, loss: %f, acc: %f, pplx: %f, lr: %.8f, cnn_lr: %.8f, finetune: %s, optim: %s, %.3f', 
      iter, number_of_batches, epoch,
      losses.total_loss, losses.precision[2], losses.total_perplexity,
      learning_rate, cnn_learning_rate, 
      tostring(finetune_cnn), opt.optim, elapsed_trn)))
  end

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
    logger_trn:add{
      ['time'] = elapsed_trn,
      ['iter'] = iter,
      ['epoch'] = epoch,
      ['loss'] = losses.total_loss,
    }

    local start_tst = tm:time().real
    -- evaluate the validation performance
    local val_loss, val_predictions, lang_stats, perplexity, val_precision = 
      eval_split('val', 
                 {val_images_use = opt.val_images_use, 
                  use_bn = opt.use_bn, 
                  iter = iter, 
                  checkpoint_path = opt.checkpoint_path})
    local elapsed_tst = tm:time().real
    print(string.format('validation loss: %f, perplexity: %f, precision: %f', 
                        val_loss, 
                        perplexity, 
                        val_precision))
    io.flush()
    --print(lang_stats)

    local checkpoint_path = 
      path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_predictions = val_predictions

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    io.flush(print('wrote json checkpoint to ' .. checkpoint_path .. '.json'))

    -- write the full model checkpoint as well if we did better than ever
    local current_score
    if lang_stats then
      -- use CIDEr score for deciding how well we did
      current_score = lang_stats['CIDEr']
    else
      -- use the (negative) validation loss as a score
      current_score = -val_loss
    end
    if best_score == nil or current_score > best_score then
      best_score = current_score
      if iter > 0 then -- dont save on very first iteration
        -- include the protos (which have weights) and save to file
        local save_protos = {}
        save_protos.lm = thin_lm -- these are shared clones, and point to correct param storage
        save_protos.cnn = thin_cnn
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        checkpoint.vocab = loader:getVocab()
        torch.save(checkpoint_path .. '.t7', checkpoint)
        io.flush(print( 'wrote checkpoint to ' .. checkpoint_path .. '.t7'))
      end
    end
    if lang_stats then
      logger_tst:add{
        ['time'] = elapsed_tst,
        ['iter'] = iter,
        ['epoch']= epoch,
        ['loss'] = val_loss,
        ['CIDEr']  = lang_stats['CIDEr'],
        ['ROUGE_L']= lang_stats['ROUGE_L'],
        ['METEOR'] = lang_stats['METEOR'],
        ['Bleu_1'] = lang_stats['Bleu_1'],
        ['Bleu_2'] = lang_stats['Bleu_2'],
        ['Bleu_3'] = lang_stats['Bleu_3'],
        ['Bleu_4'] = lang_stats['Bleu_4'],
      }
    else
      logger_tst:add{
        ['time'] = elapsed_tst,
        ['iter'] = iter,
        ['epoch']= epoch,
        ['loss'] = val_loss,
      }
    end
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 20 then
    io.flush(print( 'loss seems to be exploding, quitting.'))
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion

end

