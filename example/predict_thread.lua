require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'models.LanguageModel'
local net_utils = require 'misc.net_utils'

torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

local model_filename = 
  '/storage/coco/checkpoints/_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5/model_id_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5.t7'
local batch_size = checkpoint.opt.batch_size
local checkpoint = torch.load(opt.model)
local opt = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model', 'seq_per_img', 'image_size', 'crop_size'}
for k,v in pairs(opt) do 
  opt[v] = checkpoint.opt[v]
end
local vocab = checkpoint.vocab

local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.lm:createClones()
for k,v in pairs(protos) do v:cuda() end
protos.cnn:evaluate()
protos.lm:evaluate()

local nThreads = 4
local donekys = Threads(
  nThreads,
  function() end,
  function(thread_index)
    local tid = thread_index
    print(('===> Starting donkey with id: %d seed: %d'):format(tid, seed))
    loader = paths.dofile('../donkey/test_donkey.lua')
  end
)
  

local testBatch = function(n, batch, labels)
  local batch_processed = batch[{1},{},{},{}]
  local feats = protos.cnn:forward(batch_processed)
  local sample_opts = 
    { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
  local seq = protos.lm:sample(feats, sample_opts)
  local sents = net_utils.decode_sequence(vocab, seq)
  print(sents[1])
end

