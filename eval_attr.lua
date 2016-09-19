require 'paths'
require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = false

local agent_path = '/works/vision_language'
package.path = paths.concat(agent_path, '?.lua;') .. package.path
print(string.format('agent path: %s', agent_path))
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'models.LanguageModel'
require 'models.FeatExpander'
local net_utils = require 'misc.net_utils'

local model_filename = 
  '*.t7'
local image_size = 342
local crop_size = 299
local seq_length = 14
local batch_size = 1

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Testing an attribute-sequence model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model', model_filename, 'path to model to evaluate')
-- Basic options
cmd:option('-image_size', image_size, 'size of input image')
cmd:option('-crop_size', crop_size, 'size of input image')
cmd:option('-seq_length', seq_length, 'max. length of a sentence')
cmd:option('-batch_size', batch_size, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', -1, 'how many images to use when periodically evaluating the loss? (-1 = all)')
--cmd:option('-dump_path', 0, 'Write image paths along with predictions into vis json? (1=yes,0=no)')

-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 1, 
  'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 
  'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

cmd:option('-input_h5',
  '/storage/freebee/tshirts_shirts_blous_knit_jacket_onepiece_skirts_coat_cardigan_vest_pants_leggings_shoes_bags_swimwears_hat_panties_bra.image_sentence.txt.shuffle.txt.cutoff50.h5',
  'path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json',
  '/storage/freebee/tshirts_shirts_blous_knit_jacket_onepiece_skirts_coat_cardigan_vest_pants_leggings_shoes_bags_swimwears_hat_panties_bra.image_sentence.txt.shuffle.txt.cutoff50.json',
  'path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
cmd:option('-output',
  '',
  'path to the write prediction-gt pairs in disk')
cmd:option('-split', 
  'test', 
  --'val', 
  --'train',
  'if running on images, which split to use: val|test|train')

-- misc
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.manualSeed(opt.seed)

assert(opt.output, 'use -output')
logger_output = io.open(opt.output, 'w')

assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
opt.seq_per_img = checkpoint.opt.seq_per_img
local vocab = checkpoint.vocab 
--print(vocab)


local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}

local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.crit = nn.LanguageModelCriterion()
protos.lm:createClones()
for k,v in pairs(protos) do v:cuda() end
-- TODO: doesnt work on cudnn-v5. Oops!!
--protos.cnn = paths.dofile('/works/image-encoder/utils/BN-absorber.lua')(protos.cnn)
protos.cnn:evaluate()
protos.lm:evaluate()
collectgarbage()


local function eval_split(split, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local num_images = utils.getopt(evalopt, 'num_images', true)

  -- rewind iteator back to first datapoint in the split
  loader:resetIterator(split)
  local n = 0
  local loss_sum = 0
  while true do
    -- fetch a batch of data
    local data = loader:getBatch{
      batch_size = opt.batch_size, 
      image_size = opt.image_size, 
      split = split, 
      seq_per_img = opt.seq_per_img
    }

    data.labels = data.labels[{{1,opt.seq_length},{}}]

    data.images = net_utils.preprocess(
      data.images, opt.crop_size, false, false
    )
    n = n + data.images:size(1)

    -- forward the model to get loss
    local feats = protos.cnn:forward(data.images)
    if feats:dim() == 1 then
      local feats_2d = torch.CudaTensor(1,feats:size(1)) 
      feats_2d[{1,{}}] = feats:clone()
      feats = feats_2d
    end

    -- evaluate loss if we have the labels
    local loss = 0
    local expanded_feats = protos.expander:forward(feats)
    local logprobs = protos.lm:forward{expanded_feats, data.labels}
    -- logprobs:size() --> (opt.seq_length+2, opt.batch_size, # of total words)
    loss = protos.crit:forward(logprobs, data.labels)
    loss_sum = loss_sum + loss

    -- forward the model to also get generated samples for each image
    local sample_opts = {sample_max = opt.sample_max, 
                         beam_size = opt.beam_size, 
                         temperature = opt.temperature}
    local seq = protos.lm:sample(feats, sample_opts)
    local sents, num_words = net_utils.decode_sequence(vocab, seq)

    local gt_sents = ''
    for i=1,opt.seq_length do
      if vocab[tostring(data.labels[i][1])] ~= nil then
        gt_sents = 
          string.format('%s %s', gt_sents, vocab[tostring(data.labels[i][1])])
      end
    end

    for k=1,#sents do
      local entry = {
        image_id = data.infos[k].id, 
        file_path = data.infos[k].file_path, 
        caption = sents[k], 
        gt = gt_sents}
      if verbose then
        print(string.format('image %s: %s predicted: %s',entry.image_id, entry.file_path, entry.caption))
        print(string.format('image %s: %s gt:       %s', entry.image_id, entry.file_path, entry.gt))
      end
      logger_output:write(string.format('%s\t%s\n', entry.caption, entry.gt))
      io.flush()
    end

    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, num_images)
    if verbose then
      print(string.format('evaluating performance... %d/%d (loss: %f)', 
        ix0-1, ix1, loss))
    end

    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if num_images >= 0 and n >= num_images then break end -- we've used enough images
  end

  if n % opt.batch_size * 100 == 0 then collectgarbage() end

  return loss_sum/n
end

local loss = eval_split(opt.split, {num_images = opt.num_images})
print(string.format('loss: %f', loss))

logger_output:close()
print('Done')
