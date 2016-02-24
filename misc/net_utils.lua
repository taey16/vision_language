require 'nn'
require 'cunn'
require 'cudnn'

local utils = require 'misc.utils'
--local image_utils = require 'misc.image_utils' 
local net_utils = {}

net_utils.cnn_model_mean = 
  -- for inception7~ResCeption
  --torch.FloatTensor{0.48429165393391, 0.45580376382619, 0.40397758524087}
  -- for inception-v3-2015-12-05
  torch.FloatTensor{0.4853717905167, 0.45622173301884, 0.4061366788954}
net_utils.cnn_model_std = 
  -- for inception7~ResCeption
  --torch.FloatTensor{0.22523080791307, 0.22056471186989, 0.22048053881112}
  -- for inception-v3-2015-12-05
  torch.FloatTensor{0.22682182875849, 0.22206057852892, 0.22145828935297}


function net_utils.build_cnn(opt)
  local model_filename = utils.getopt(opt, 'model_filename', 
    '/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_40.bn_removed.t7')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  local original_model = torch.load(model_filename)
  local vision_encoder = original_model:get(1)
  --local vision_encoder = original_model:get(1):get(1)
  local cnn_part = nn.Sequential()
  cnn_part:add(vision_encoder)
  cnn_part:add(nn.View(2048))
  print(cnn_part)
  print(string.format('===> Loading pre-trained model complete', model_filename))
  return cnn_part 
end


function net_utils.preprocess_for_predict(imgs, crop_size, data_augment, on_gpu)
  -- used ofr 3d tensor input
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')
  local h,w = imgs:size(2), imgs:size(3)
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
    imgs = imgs[{ {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end
  for c=1,3 do
    imgs[{{c},{},{}}]:add(-net_utils.cnn_model_mean[c])
    imgs[{{c},{},{}}]:div(net_utils.cnn_model_std[c])
  end
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end
  return imgs
end


function net_utils.preprocess(imgs, crop_size, data_augment, on_gpu)
  -- used ofr 3d tensor input
  assert(data_augment ~= nil, 'pass this in. careful here.')
  assert(on_gpu ~= nil, 'pass this in. careful here.')
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
  --imgs = image_utils.random_flip(imgs)
  imgs = torch.div(imgs:float(), 255.0)
  for c=1,3 do
    imgs[{{},{c},{},{}}]:add(-net_utils.cnn_model_mean[c])
    imgs[{{},{c},{},{}}]:div(net_utils.cnn_model_std[c])
  end
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end
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
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
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

return net_utils


