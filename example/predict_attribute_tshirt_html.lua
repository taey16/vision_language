require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true
cudnn.verbose = false
package.path = '/works/demon_11st/lua/?.lua;' .. package.path
local agent = require 'agent.agent_attribute'
local demon_utils = require 'utils.demon_utils'

local agent_filename = agent.model_filename
local opts = agent.opts
local sample_opts = agent.sample_opts
agent_filename = string.split(agent_filename, '/')
local output_dic_filename = string.format(
  '11st_julia_tshirts_shirtstes_blous_knit_sentences.%s.txt', agent_filename[#agent_filename]
  --'11st_julia_tshirts_shirtstes_blous_sentences_from99502.%s.txt', agent_filename[#agent_filename]
  --'11st_julia_tshirts_shirtstes_blous_sentences_from75001.%s.txt', agent_filename[#agent_filename]
  --'11st_julia_tshirts_shirtstes_blous_sentences.%s.txt', agent_filename[#agent_filename]
)
local outfile_dic = io.open(output_dic_filename, 'w')
local output_html_filename = 
  string.format('%s.html', output_dic_filename)

local input_list = 
  '/storage/attribute/PBrain_11st_julia_tshirts_shirts_blous_knit_seed123_limit300000.csv'
  --'/storage/attribute/PBrain_11st_julia_tshirts_shirts_blous_seed123_limit99501_300000.csv'
  --'/storage/attribute/PBrain_11st_julia_tshirts_shirts_blous_seed123_limit75000_300000.csv'
  --'/storage/attribute/PBrain_11st_julia_tshirts_shirts_blous_seed123_limit300000.csv'
local url_list = demon_utils.load_list(input_list)

local fp_html = io.open(output_html_filename, 'w')
fp_html:write("<html>\n  <head>\n    <table>\n      <tr>\n")

for iter, url in pairs(url_list) do
  local image_url = string.format('http://i.011st.com/%s', url)
  local image_file= demon_utils.download_image(image_url)
  io.flush(print(iter .. ' ' .. image_url))

  local sents = agent.get_attribute(image_file)
  if sents then
    outfile_dic:write(string.format('%s,%s\n', image_url, sents[1]))
    print(sents[1])
    os.execute(('rm -f %s'):format(image_file))

    fp_html:write(string.format("        <td><img src=\"%s\" height=\"292\" width=\"292\"></br>\n", image_url))
    fp_html:write(string.format("        <font color=\"green\">%s</font>", sents[1]))
    fp_html:write("      </td>\n")
    fp_html:write("      </td>\n")
    if iter % 5 == 0 then
      fp_html:write("    </tr>\n<tr>\n")
    end
  end
end

outfile_dic:close()
fp_html:close()
io.flush(print('Done'))

