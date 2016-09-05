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
local coco_utils = paths.dofile('../misc/coco_utils.lua')


local model_filename = 
  '/storage/coco/checkpoints/coco_123287_3200_seq_length-1/inception-v3-2015-12-05_embed__bs16_flipfalse_croptrue_original_init_gamma0.100000_lstm_tanh_hid384_lay3_drop5.000000e-01_adam_lr4.000000e-04_seed0.50_start300000_every50000_finetune0_cnnlr4.000000e-04_cnnwc1.000000e-07_tsne/model_idinception-v3-2015-12-05_embed__bs16_flipfalse_croptrue_original_init_gamma0.100000_lstm_tanh_hid384_lay3_drop5.000000e-01_adam_lr4.000000e-04_seed0.50_start300000_every50000_finetune0_cnnlr4.000000e-04_cnnwc1.000000e-07_tsne.t7'
  --'/storage/coco/checkpoints/_ReCept_bn_removed_epoch35_bs16_embedding2048_encode384_layer2_lr4e-4/model_id_ReCept_bn_removed_epoch35_bs16_embedding2048_encode384_layer2_lr4e-4.t7'
  --'/storage/coco/checkpoints/_ReCept_bn_removed_epoch35_bs16_embedding2048_encode256_layer2_lr4e-4/model_id_ReCept_bn_removed_epoch35_bs16_embedding2048_encode256_layer2_lr4e-4.t7'
  --'/storage/coco/checkpoints/_inception7_bs16_encode256_layer2/model_id_inception7_bs16_encode256_layer2.t7'
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
local opt
for k,v in pairs(opt) do 
  opt[v] = checkpoint.opt[v]
end
local vocab = checkpoint.vocab
local sample_opts = { 
  sample_max = opt.sample_max, 
  beam_size = 1, 
  temperature = opt.temperature 
}

local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img)
protos.lm:createClones()
for k,v in pairs(protos) do v:cuda() end
protos.cnn:evaluate()
protos.lm:evaluate()

local output_dic_filename = 
  'COCO_test_sentences.txt'
local outfile_dic = io.open(output_dic_filename, 'w')

torch.manualSeed(999)
filename_list, url_list, list_table = coco_utils.get_test()
info = coco_utils.permute(list_table)

local iter = 1
for k, v in pairs(info) do
  local fname = string.format('%s/%s', '/storage/coco/test2015', k)
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

outfile_dic:close()
