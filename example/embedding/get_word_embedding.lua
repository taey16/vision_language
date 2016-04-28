
--- file to perform some analysis

require 'cutorch'
require 'cunn'
require 'cudnn'
local agent_path = '/works/vision_language'
package.path = paths.concat(agent_path, '?.lua;') .. package.path
print(string.format('agent path: %s', agent_path))
local utils = require 'misc.utils'
require 'misc.DataLoader'
require 'misc.DataLoaderRaw'
require 'models.LanguageModel'
require 'models.FeatExpander'
local net_utils = require 'misc.net_utils'
local manifold = require 'manifold'
torch.setdefaulttensortype('torch.DoubleTensor')

local model_filename =
  -- 18cate
  '/storage/attribute/checkpoints/tshirts_shirts_blous_knit_jacket_onepiece_skirts_coat_cardigan_vest_pants_leggings_shoes_bags_swimwears_hat_panties_bra_801544_40000_seq_length14/resception_ep29_bs16_flipfalse_croptrue_original_init_gamma0.100000_lstm_tanh_hid512_lay2_drop2.000000e-01_adam_lr1.000000e-03_seed0.90_start541152_every45096_finetune0_cnnlr1.000000e-03_cnnwc1.000000e-05_retrain_iter0/model_idresception_ep29_bs16_flipfalse_croptrue_original_init_gamma0.100000_lstm_tanh_hid512_lay2_drop2.000000e-01_adam_lr1.000000e-03_seed0.90_start541152_every45096_finetune0_cnnlr1.000000e-03_cnnwc1.000000e-05_retrain_iter0.t7'
local image_size = 342
local crop_size = 299
local seq_length = 14
local batch_size = 1

local protos = torch.load(model_filename)
local model = protos.protos
local vocab = protos.vocab
local embedding_weight = model.lm.lookup_table.weight:float()
local num_words = embedding_weight:size(1)
local embedding_size = embedding_weight:size(2)
print(vocab)

local embedding_input = torch.FloatTensor(1, num_words)
embedding_input[1] = torch.range(1, num_words)
local embedding_output = model.lm.lookup_table:forward(embedding_input:cuda()):squeeze()
print(embedding_output:size())

local word_vector = {}
for i, val in pairs(vocab) do
  word_vector[val] = embedding_output[{{tonumber(i)}, {}}]:squeeze()
end

--normalize
for i, val in pairs(word_vector) do
	local norm = word_vector[i]:norm()
	if norm > 0 then
		word_vector[i]:div(norm)
	end
end


function dot(a, b)
	return torch.dot(word_vector[a], word_vector[b])
end


function nearest_neighbors()
	for i, v in pairs(word_vector) do
		local maxDot = -10
		local NN = i
		for j, w in pairs(word_vector) do
			if j ~= i then
				if torch.dot(v,w) > maxDot then
					maxDot = torch.dot(v,w)
					NN = j
				end
			end
		end
		print(i, NN ,maxDot)
	end
end


function find_len(table)
	local cnt = 0
	for k, v in pairs(table) do
		cnt = cnt+1
	end
	return cnt
end


function tsne(vec)
	--local num_words = find_len(vec)
  print('num_words: ' .. num_words)
  print('embedding_size: ' .. embedding_size)
	local m = torch.zeros(num_words, embedding_size)
	local i = 1
	local symbols = {}
	for k, val in pairs(vec) do
    --print(vec[k]:size())
    --print(m[i]:size())
    m[i]:copy(vec[k])
    symbols[i] = k
    --print(m[i])
    --print(string.format('symbol: %s, vec[1] = %f', symbols[i], m[i][1]))
    i = i + 1
	end
  tsne_opts = {ndims = 2, perplexity = 50, pca = 256, use_bh = false}
  --tsne_opts = {ndims = 2, perplexity = 50, use_bh = false}
  mapped_x1 = manifold.embedding.tsne(m, tsne_opts)
  return mapped_x1, symbols
end

tsne, symbols = tsne(word_vector)

--write
local file = io.open('tsne.txt', "w");
for i=1, #symbols do
	file:write(symbols[i] .. ' ' .. tsne[i][1]  .. ' ' .. tsne[i][2] .. '\n')
end
