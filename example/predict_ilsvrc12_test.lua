require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
local cjson = require 'cjson'
package.path = '../?.lua;' .. package.path
require 'misc.DataLoaderRaw'
require 'models.LanguageModel'
local net_utils = require 'misc.net_utils'
local imagenet_utils = paths.dofile('../misc/imagenet_utils.lua')


local model_filename = 
  '/storage/coco/checkpoints/_inception7_bs16_encode256_layer2/model_id_inception7_bs16_encode256_layer2.t7'
  --'/storage/coco/checkpoints/_inception7_bs16_encode512/model_id_inception7_bs16_encode512.t7'
  --'/storage/coco/checkpoints/_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5/model_id_inception7_bs16_encode512_finetune_lr4e-6_clr1e-7_wc2e-5.t7'
local checkpoint = torch.load(model_filename)
local batch_size = checkpoint.opt.batch_size
local opt = {
  'rnn_size', 
  'input_encoding_size', 
  'drop_prob_lm', 
  'cnn_proto', 
  'cnn_model', 
  'seq_per_img', 
  'image_size', 
  'crop_size'
}
for k,v in pairs(opt) do 
  opt[v] = checkpoint.opt[v]
end
local vocab = checkpoint.vocab
local sample_opts = { 
  sample_max = opt.sample_max, 
  beam_size = 4, 
  temperature = opt.temperature 
}

local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.lm:createClones()
for k,v in pairs(protos) do v:cuda() end
protos.cnn:evaluate()
protos.lm:evaluate()

local output_dic_filename = 
  'ILSVRC2012_test_sentence.txt'
local outfile_dic = io.open(output_dic_filename, 'w')

torch.manualSeed(999)
filename_list, url_list, info = imagenet_utils.get_test()

local iter = 1
for k, v in pairs(info) do
  local fname = string.format('%s/%s', '/storage/ImageNet/ILSVRC2012/test', k)
  local url = v
  io.flush(print(iter .. ' ' .. fname))
  local img = image.load(fname)
  img = image.scale(img, opt.image_size, opt.image_size)
  if img:size(1) == 1 then
    img = img:view(1,img:size(2), img:size(3)):repeatTensor(3,1,1)
  end
  img = net_utils.preprocess_inception7_predict(img, opt.crop_size, false, 1)
  local data = torch.CudaTensor(2, 3, opt.crop_size, opt.crop_size):fill(0)
  data[{{1},{},{},{}}] = img
  local feats = protos.cnn:forward(data)
  local seq = protos.lm:sample(feats, sample_opts)
  local sents = net_utils.decode_sequence(vocab, seq)
  outfile_dic:write(string.format('%s,%s\n', url, sents[1]))
  print(sents[1])
  iter = iter+1
end

