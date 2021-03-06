require 'paths'
require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = true

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
  -- 18cate
  '/storage/attribute/checkpoints/tshirts_shirts_blous_knit_jacket_onepiece_skirts_coat_cardigan_vest_pants_leggings_shoes_bags_swimwears_hat_panties_bra_801544_40000_seq_length14/resception_ep29_bs16_flipfalse_croptrue_original_init_gamma0.100000_lstm_tanh_hid512_lay2_drop2.000000e-01_adam_lr1.000000e-03_seed0.90_start541152_every45096_finetune0_cnnlr1.000000e-03_cnnwc1.000000e-05_retrain_iter0/model_idresception_ep29_bs16_flipfalse_croptrue_original_init_gamma0.100000_lstm_tanh_hid512_lay2_drop2.000000e-01_adam_lr1.000000e-03_seed0.90_start541152_every45096_finetune0_cnnlr1.000000e-03_cnnwc1.000000e-05_retrain_iter0.t7'
local image_size = 342
local crop_size = 299
local seq_length = 14
local batch_size = 1

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Testing an Image Captioning model')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model', model_filename, 'path to model to evaluate')
-- Basic options
cmd:option('-image_size', image_size, 'size of input image')
cmd:option('-crop_size', crop_size, 'size of input image')
cmd:option('-seq_length', seq_length, 'max. length of a sentence')
cmd:option('-batch_size', batch_size, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-num_images', 
  40000, 'how many images to use when periodically evaluating the loss? (-1 = all)')
cmd:option('-language_eval', 0, 
  'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-dump_images', 0, 'Dump images into vis/imgs folder for vis? (1=yes,0=no)')
cmd:option('-dump_json', 0, 'Dump json with predictions into vis folder? (1=yes,0=no)')
cmd:option('-dump_path', 0, 'Write image paths along with predictions into vis json? (1=yes,0=no)')

-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 1, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')

-- For evaluation on a folder of images:
cmd:option('-image_folder', '', 'If this is nonempty then will predict on the images in this folder path')
cmd:option('-image_root', '', 'In case the image paths have to be preprended with a root path to an image folder')

-- For evaluation on MSCOCO images from some split:
cmd:option('-input_h5','','path to the h5file containing the preprocessed dataset. empty = fetch from model checkpoint.')
cmd:option('-input_json','','path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
cmd:option('-split', 'test', 'if running on MSCOCO images, which split to use: val|test|train')
cmd:option('-coco_json', '', 'if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')

-- misc
cmd:option('-id', 'evalscript', 'an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:text()


local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.manualSeed(opt.seed)


assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
-- override and collect parameters
if string.len(opt.input_h5) == 0 then opt.input_h5 = checkpoint.opt.input_h5 end
if string.len(opt.input_json) == 0 then opt.input_json = checkpoint.opt.input_json end
if opt.batch_size == 0 then opt.batch_size = checkpoint.opt.batch_size end
opt.seq_per_img = checkpoint.opt.seq_per_img
local vocab = checkpoint.vocab 
print(vocab)


local loader
if string.len(opt.image_folder) == 0 then
  loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json}
else
  loader = DataLoaderRaw{folder_path = opt.image_folder, coco_json = opt.coco_json}
end


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


local function eval_split(split, evalopt, fp_html)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local num_images = utils.getopt(evalopt, 'num_images', true)

  -- rewind iteator back to first datapoint in the split
  loader:resetIterator(split)
  local n = 0
  local loss_sum = 0
  local accuracy = 0
  local perplexity = 0
  local loss_evals = 0
  local predictions = {}
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
    --[[
    data.images = net_utils.preprocess_for_predict_aug5(
      data.images:squeeze(), opt.crop_size
    )
    --]]
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
    local acc,pplx = 0, 0
    local acc_per_seq = 0
    if data.labels then
      local expanded_feats = protos.expander:forward(feats)
      local logprobs = protos.lm:forward{expanded_feats, data.labels}
      -- logprobs:size() --> (opt.seq_length+2, opt.batch_size, # of total words)
      loss = protos.crit:forward(logprobs, data.labels)
      loss_sum = loss_sum + loss

      acc, pplx = protos.crit:accuracy(logprobs, data.labels)
      for i=1,opt.seq_length do 
      --for i=2,2 do 
        acc_per_seq = acc_per_seq + acc[i] 
      end
      --acc_per_seq = acc_per_seq / opt.seq_length
      --avg_acc_per_seq = avg_acc_per_seq / 1
      --accuracy = accuracy + acc[2]
      accuracy = accuracy + acc_per_seq
      perplexity = perplexity + pplx

      loss_evals = loss_evals + 1
    end

    -- forward the model to also get generated samples for each image
    local sample_opts = { 
      sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
    local seq = protos.lm:sample(feats, sample_opts)
    local sents, num_words = net_utils.decode_sequence(vocab, seq)
    acc_per_seq = (acc_per_seq-1) / num_words

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
      if opt.dump_path == 1 then
        entry.file_name = data.infos[k].file_path
      end
      table.insert(predictions, entry)
      if opt.dump_images == 1 then
        -- dump the raw image to vis/ folder
        local cmd = 'cp "' .. path.join(opt.image_root, data.infos[k].file_path) .. '" vis/imgs/img' .. #predictions .. '.jpg' -- bit gross
        print(cmd)
        os.execute(cmd) -- dont think there is cleaner way in Lua
      end
      if verbose then
        print(string.format('image %s: %s predicted: %s',entry.image_id, entry.file_path, entry.caption))
        print(string.format('image %s: %s gt:       %s', entry.image_id, entry.file_path, entry.gt))
      end
      local image_url = entry.file_path:gsub('/data2/freebee/Images', 'http://10.202.35.87/freebee')
      image_url = entry.file_path:gsub( "/data3/11st/imgdata/", "http://10.202.35.87/freebee3/")
      fp_html:write(string.format("        <td><img src=\"%s\" height=\"299\" width=\"299\"></br>\n", image_url))
      fp_html:write(string.format("        <font color=\"black\" size=2>%s</font></br>",entry.caption ))
      fp_html:write(string.format("        <font color=\"green\" size=2>%s</font>", entry.gt))
      fp_html:write("      </td>\n")
      fp_html:write("      </td>\n")
      if n % 5 == 0 then
        fp_html:write("    </tr>\n<tr>\n")
      end
    end


    -- if we wrapped around the split or used up val imgs budget then bail
    local ix0 = data.bounds.it_pos_now
    local ix1 = math.min(data.bounds.it_max, num_images)
    if verbose then
      print(string.format('evaluating performance... %d/%d (loss: %f, precision: %f, pplx: %f)', 
        ix0-1, ix1, 
        loss, acc_per_seq, pplx))
    end

    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if num_images >= 0 and n >= num_images then break end -- we've used enough images
  end

  local lang_stats
  if opt.language_eval == 1 then
    lang_stats = net_utils.language_eval(predictions, opt.id)
  end

  if n % opt.batch_size * 100 == 0 then collectgarbage() end

  return loss_sum/loss_evals, predictions, lang_stats, perplexity/loss_evals, accuracy/loss_evals
end

local fp_html = io.open('./cate18.eval.html', 'w')
fp_html:write("<html>\n  <head>\n    <table>\n      <tr>\n")

local loss, split_predictions, lang_stats, perplexity, accuracy = eval_split(opt.split, {num_images = opt.num_images}, fp_html)
print(string.format('loss: %f, acc: %f, pplx: %f', loss, accuracy, perplexity))
if lang_stats then
  print(lang_stats)
end

fp_html:close()

if opt.dump_json == 1 then
  -- dump the json
  utils.write_json('vis/vis.json', split_predictions)
end

