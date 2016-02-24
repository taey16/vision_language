require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = true
require 'nngraph'
local path = require 'pl.path'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
require 'misc.DataLoader'
require 'misc.optim_updates'
require 'models.LanguageModel'
require 'models.FeatExpander'
require 'optim'
require 'cephes' -- for cephes.log2


local opt = paths.dofile('opts/opt_attribute_tshirts_shirts_blous_knit_inception-v3.lua')
--local opt = paths.dofile('opts/opt_attribute_tshirts_shirts_blous_inception-v3.lua')
--local opt = paths.dofile('opts/opt_attribute_tshirts_shirts_inception-v3.lua')
--local opt = paths.dofile('opts/opt_attribute_tshirts_inception-v3.lua')
--local opt = paths.dofile('opts/opt_coco_inception-v3.lua')
--local opt = paths.dofile('opts/opt_coco_inception7.lua')
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.manualSeedAll(opt.seed)

local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

local protos = {}
if string.len(opt.start_from) > 0 then
  -- load protos from file
  io.flush(print('initializing weights from ' .. opt.start_from))
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  net_utils.unsanitize_gradients(protos.cnn)
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually
  protos.expander = nn.FeatExpander(opt.seq_per_img) -- not in checkpoints, create manually
  cudnn.convert(protos.cnn, cudnn)
  print(protos.cnn)
  print(protos.lm)
  print(protos.expander)
  print(protos.crit)
else
  -- create protos from scratch
  -- intialize language model
  local lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = opt.num_rnn_layers
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size * opt.seq_per_img
  lmOpt.lstm_activation = opt.lstm_activation
  lmOpt.rnn_type = opt.rnn_type
  protos.lm = nn.LanguageModel(lmOpt)
  -- initialize the ConvNet
  protos.cnn = net_utils.build_cnn(
    {encoding_size = opt.input_encoding_size, model_filename = opt.torch_model}
  )
  -- initialize a special FeatExpander module that "corrects" for the batch number discrepancy 
  -- where we have multiple captions per one image in a batch. This is done for efficiency
  -- because doing a CNN forward pass is expensive. We expand out the CNN features for each sentence
  protos.expander = nn.FeatExpander(opt.seq_per_img)
  -- criterion for the language model
  protos.crit = nn.LanguageModelCriterion()
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

-- flatten and prepare all model parameters to a single vector. 
-- Keep CNN params separate in case we want to try to get fancy with different optims on LM/CNN
local params, grad_params = protos.lm:getParameters()
local cnn_params, cnn_grad_params = protos.cnn:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN: ', cnn_params:nElement())
assert(params:nElement() == grad_params:nElement())
assert(cnn_params:nElement() == cnn_grad_params:nElement())

-- construct thin module clones that share parameters with the actual
-- modules. These thin module will have no intermediates and will be used
-- for checkpointing to write significantly smaller checkpoint files
local thin_lm = protos.lm:clone()
-- TODO: we are assuming that LM has specific members! figure out clean way to get rid of, not modular.
thin_lm.core:share(protos.lm.core, 'weight', 'bias')
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
local thin_cnn = protos.cnn:clone('weight', 'bias')
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

  protos.cnn:evaluate()
  protos.lm:evaluate()
  -- rewind iteator back to first datapoint in the split
  loader:resetIterator(split)
  local n = 0
  local loss_sum = 0
  local logprobs_sum = 0
  local perplexity = 0
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

    -- preprocess in place, and don't augment
    data.images = net_utils.preprocess(
      data.images, opt.crop_size, false, opt.gpuid >= 0
    )
    n = n + data.images:size(1)

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    local expanded_feats = protos.expander:forward(feats)
    local logprobs = protos.lm:forward{expanded_feats, data.labels}
    local loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss
    --logprobs_sum = logprobs_sum + logprobs
    loss_evals = loss_evals + 1

    -- forward the model to also get generated samples for each image
    local seq = protos.lm:sample(feats)
    local sents = net_utils.decode_sequence(vocab, seq)
    for k=1,#sents do
      local entry = {image_id = data.infos[k].id, file_path = data.infos[k].file_path, caption = sents[k]}
      table.insert(predictions, entry)
      if verbose then
        io.flush(print(string.format(
          'image %s(%s): %s', entry.image_id, entry.file_path, entry.caption
        )))
      end
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, val_images_use)
    if verbose then
      io.flush(print(string.format(
        'evaluating validation performance... %d/%d (%f)', ix0-1, ix1, loss
      )))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use then break end -- we've used enough images
  end

  --perplexity = -cephes.log2(logprobs_sum)
  --perplexity = cephes.pow(2.0, perplexity / loss_evals)

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  return loss_sum/loss_evals, predictions, lang_stats, perplexity
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
  -- preproces in-place, data augment in training
  data.images = net_utils.preprocess(
    data.images, opt.crop_size, true, opt.gpuid >= 0
  )

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
  --local perplexity = cephes.pow(2.0, -cephes.log2(logprobs) / opt.batch_size)
  local perplexity = 0
  
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(logprobs, data.labels)
  -- backprop language model
  local dexpanded_feats, ddummy = 
    unpack(protos.lm:backward({expanded_feats, data.labels}, dlogprobs))
  -- backprop the CNN, but only if we are finetuning
  if finetuen_cnn then
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

  -- and lets get out!
  local losses = { total_loss = loss, total_perplexity = perplexity }
  return losses
end

local logger_trn = 
  optim.Logger(paths.concat(opt.checkpoint_path, 'train.log'))
local logger_tst = 
  optim.Logger(paths.concat(opt.checkpoint_path, 'test.log'))


-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local iter = 0
local loss0
local number_of_batches = opt.train_samples / opt.batch_size
local optim_state = {}
local cnn_optim_state = {}
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local best_score
local tm = torch.Timer()

while true do  
  local start_trn = tm:time().real

  local finetune_cnn = false
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    finetune_cnn = true
  end

  -- eval loss/gradient
  local losses = lossFun(finetune)
  if iter % opt.losses_log_every == 0 then 
    loss_history[iter] = losses.total_loss 
  end

  -- decay the learning rate for both LM and CNN
  local learning_rate = opt.learning_rate
  local cnn_learning_rate = opt.cnn_learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
    cnn_learning_rate = cnn_learning_rate * decay_factor
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- do a cnn update (if finetuning, and if rnn above us is not warming up right now)
  if finetune_cnn then
    if opt.cnn_optim == 'sgd' then
      sgd(cnn_params, cnn_grad_params, cnn_learning_rate)
    elseif opt.cnn_optim == 'sgdm' then
      sgdm(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn_optim_state)
    elseif opt.cnn_optim == 'adam' then
      adam(cnn_params, cnn_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn_optim_state)
    else
      error('bad option for opt.cnn_optim')
    end
  end

  local elapsed_trn = tm:time().real - start_trn
  local epoch = iter * 1.0 / number_of_batches
  if iter % opt.display == 0 then
    io.flush(print(string.format(
      '%d/%d: %.2f, trn loss: %f, pplx: %f, lr: %.8f, cnn_lr: %.8f, finetune: %s, optim: %s, %.3f', 
      iter, number_of_batches, epoch,
      losses.total_loss, losses.total_perplexity,
      learning_rate, cnn_learning_rate, 
      tostring(finetune_cnn), opt.optim, elapsed_trn
    )))
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
    local val_loss, val_predictions, lang_stats, perplexity = 
      eval_split('val', {val_images_use = opt.val_images_use})
    local elapsed_tst = tm:time().real
    io.flush( print(string.format(
        'validation loss: %f, perplexity: %f', val_loss, perplexity
    )))
    --print(lang_stats)
    val_loss_history[iter] = val_loss
    if lang_stats then
      val_lang_stats_history[iter] = lang_stats
    end

    local checkpoint_path = 
      path.join(opt.checkpoint_path, 'model_id' .. opt.id)

    -- write a (thin) json report
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.val_loss_history = val_loss_history
    -- save these too for CIDEr/METEOR/etc eval
    checkpoint.val_predictions = val_predictions
    checkpoint.val_lang_stats_history = val_lang_stats_history

    utils.write_json(checkpoint_path .. '.json', checkpoint)
    io.flush(print(
      'wrote json checkpoint to ' .. checkpoint_path .. '.json'
    ))

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
        io.flush(print(
          'wrote checkpoint to ' .. checkpoint_path .. '.t7'
        ))
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

