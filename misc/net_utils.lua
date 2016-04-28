require 'nn'
require 'cunn'
require 'cudnn'

local utils = require 'misc.utils'
local image_utils = require 'misc.image_utils' 
local net_utils = {}

net_utils.cnn_model_mean = 
  -- for inception7~ResCeption
  --torch.FloatTensor{0.48429165393391, 0.45580376382619, 0.40397758524087}
  -- for inception-v3-2015-12-05, resception
  torch.FloatTensor{0.4853717905167, 0.45622173301884, 0.4061366788954}
net_utils.cnn_model_std = 
  -- for inception7~ResCeption
  --torch.FloatTensor{0.22523080791307, 0.22056471186989, 0.22048053881112}
  -- for inception-v3-2015-12-05, resception
  torch.FloatTensor{0.22682182875849, 0.22206057852892, 0.22145828935297}


function net_utils.build_cnn(opt)
  local model_filename = utils.getopt(opt, 'model_filename')
  local encoding_size = utils.getopt(opt, 'encoding_size')
  local vision_encoder = torch.load(model_filename)
  vision_encoder.modules[#vision_encoder] = nil
  vision_encoder.modules[#vision_encoder] = nil
  --[[
  local cnn_part = nn.Sequential()
  cnn_part:add(vision_encoder)
  cnn_part:add(nn.View(2048))
  print(cnn_part)
  print(string.format('===> Loading pre-trained model complete', model_filename))
  return cnn_part 
  --]]
  print(string.format('===> Loading pre-trained model complete: %s', model_filename))
  print(vision_encoder)
  return vision_encoder
end


function net_utils.preprocess_for_predict_aug5(imgs, crop_size)
  -- used ofr 3d tensor input
  assert(imgs:dim() == 3, 
    'does not support 4D images in net_utils.preprocess_for_predict')
  local h,w = imgs:size(2), imgs:size(3)
  local cnn_input_size = crop_size

  --[[
  local oH = crop_size
  local oW = crop_size
  local iH = h
  local iW = w
  local w1 = math.ceil((iW-oW)/2)
  local h1 = math.ceil((iH-oH)/2)
  local output = torch.FloatTensor(5, 3, oW, oH)
  output[{1 ,{},{},{}}] = image.crop(imgs 1,    1,    oW+1, oH+1)
  output[{2 ,{},{},{}}] = image.crop(imgs, iW-oW,1,    iH,   oH+1)
  output[{3 ,{},{},{}}] = image.crop(imgs, 1,    iH-oH,oW+1, iH)
  output[{4 ,{},{},{}}] = image.crop(imgs, iW-oW,iH-oH,iW,   iH)
  output[{5 ,{},{},{}}] = image.crop(imgs, w1,   h1,   w1+oW,h1+oH)
  --]]
  local output = image_utils.augment_image(imgs, {3, h, w}, {3, crop_size, crop_size})
  output = torch.div(output:float(), 255.0)

  for aug=1,5 do
    for c=1,3 do
      output[{{aug}, {c},{},{}}]:add(-net_utils.cnn_model_mean[c])
      output[{{aug}, {c},{},{}}]:div(net_utils.cnn_model_std[c])
    end
  end
  output = output:cuda()
  return output
end


function net_utils.preprocess_for_predict(imgs, crop_size)
  -- used ofr 3d tensor input
  assert(imgs:dim() == 3, 
    'does not support 4D images in net_utils.preprocess_for_predict')
  local h,w = imgs:size(2), imgs:size(3)
  local cnn_input_size = crop_size

  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    -- sample the center
    xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    imgs = imgs[{ {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end
  for c=1,3 do
    imgs[{{c},{},{}}]:add(-net_utils.cnn_model_mean[c])
    imgs[{{c},{},{}}]:div(net_utils.cnn_model_std[c])
  end
  imgs = imgs:cuda()
  return imgs
end


function net_utils.preprocess(imgs, crop_size, data_augment, flip_jitter)
  -- used ofr 3d tensor input
  assert(data_augment ~= nil, 'pass this in. careful here.')
  local h,w = imgs:size(3), imgs:size(4)
  local cnn_input_size = crop_size

  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    if data_augment then
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end
  if flip_jitter == true then
    imgs = image_utils.random_flip(imgs)
  end
  imgs = torch.div(imgs:float(), 255.0)
  for c=1,3 do
    imgs[{{},{c},{},{}}]:add(-net_utils.cnn_model_mean[c])
    imgs[{{},{c},{},{}}]:div(net_utils.cnn_model_std[c])
  end
  imgs = imgs:cuda()
  return imgs
end


function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end


function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. 
  -- And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end


function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end


function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end


--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  local word_count = 0
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
      word_count = word_count + 1
    end
    table.insert(out, txt)
  end
  return out, word_count
end


function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end


-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end


function net_utils.tsne_embedding(nn_embedding, vocab, opt_iter, checkpoint_path)
  local embedding_weight = nn_embedding.weight:float()
  local num_words = embedding_weight:size(1)
  local embedding_size = embedding_weight:size(2)
  local embedding_input = torch.FloatTensor(1, num_words)
  embedding_input[1] = torch.range(1, num_words)

  local embedding_output= nn_embedding:forward(embedding_input:cuda()):squeeze()

  local word_vectors = {}
  for word_idx, word in pairs(vocab) do
    word_vectors[word] = embedding_output[{{tonumber(word_idx)},{}}]:squeeze()
    local norm = word_vectors[word]:norm()
    if norm > 0 then
      word_vectors[word]:div(norm)
    end
  end
  function tsne(vec, num_words, embedding_size)
    local manifold = require 'manifold'  
    print('Start to tsne embedding ...')
    print('num_words: ' .. num_words)
    print('embedding_size: ' .. embedding_size)
    torch.setdefaulttensortype('torch.DoubleTensor')
    local tsne_input = torch.zeros(num_words, embedding_size)
    local iter = 1
    local words = {}
    -- vec: {'word0': vector, 'word1'= vector, 'word2' = vector}
    for key, val in pairs(vec) do
      tsne_input[iter]:copy(vec[key])
      words[iter] = key
      iter = iter + 1
    end
    tsne_opts = {ndims = 2, perplexity = 50, pca = 64, use_bh = false}
    local tsne_output = manifold.embedding.tsne(tsne_input, tsne_opts)
    return tsne_output, words
  end

  local tsne_output, words = tsne(word_vectors, num_words, embedding_size)

  local output_filename = string.format('%s/tsne_%08d.log', checkpoint_path, opt_iter)
  local fp = io.open(output_filename, 'w')
  for i=1, #words do
	  fp:write(words[i] .. ' ' .. tsne_output[i][1]  .. ' ' .. tsne_output[i][2] .. '\n')
  end
  fp:close()
  print('End of tsne embedding in ' .. output_filename)
  io.flush()
  torch.setdefaulttensortype('torch.FloatTensor')
end

return net_utils

