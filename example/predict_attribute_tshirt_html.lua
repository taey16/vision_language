require 'torch'
require 'hdf5'
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
local input_list_path = '/storage/attribute'
local input_list =
  '/storage/attribute/PBrain_all.csv'
local output_dic_filename = string.format(
  --'%s.image_sentence.txt', input_list
  '%s.feature.txt', input_list
)
local output_h5_filename = string.format(
  --'%s.image_sentence.txt', input_list
  '%s.feature.h5', input_list
)
local outfile_dic = io.open(paths.concat(input_list_path, output_dic_filename), 'w')
local output_html_filename = 
  paths.concat(input_list_path, string.format('%s.html', output_dic_filename))
print(output_dic_filename)
print(output_html_filename)

local url_list = demon_utils.load_list(paths.concat(input_list_path, input_list))

local fp_html = io.open(output_html_filename, 'w')
fp_html:write("<html>\n  <head>\n    <table>\n      <tr>\n")

local feature_vector = torch.FloatTensor(10000, 2048):fill(-1)
local urls = {}
local it = 1
for iter, url in pairs(url_list) do
  local image_url = string.format('http://i.011st.com/%s', url)
  local image_file= demon_utils.download_image(image_url)
  io.flush(print(iter .. ' ' .. image_url))

  local sents, probs, feature = agent.get_attribute(image_file)
  if sents then
    feature_vector[{{it},{}}] = feature:float():squeeze()
    table.insert(urls, image_url)
    it = it + 1 
    outfile_dic:write(string.format('%s,%s\n', image_url, sents[1]))
    --print(sents[1])
    os.execute(('rm -f %s'):format(image_file))

    fp_html:write(string.format("        <td><img src=\"%s\" height=\"299\" width=\"299\"></br>\n", image_url))
    fp_html:write(string.format("        <font color=\"green\">%s</font>", sents[1]))
    fp_html:write("      </td>\n")
    fp_html:write("      </td>\n")
    if iter % 5 == 0 then
      fp_html:write("    </tr>\n<tr>\n")
    end

    if it % 10000 == 0 then
      local output_h5_fp = hdf5.open(output_h5_filename, 'w')
      output_h5_fp:write('feature', feature_vector)
      --output_h5_fp:write('urls', urls)
      output_h5_fp:close()
    end
  end
end

outfile_dic:close()
fp_html:close()
io.flush(print('Done'))

