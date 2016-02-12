
require 'torch'

local coco_utils = {}

function coco_utils.permute(tab, n, count, manualSeed)
  torch.manualSeed(manualSeed)
  n = n or #tab
  for i = 1, count or n do
    local j = math.random(i, n)
    tab[i], tab[j] = tab[j], tab[i]
  end
  return tab
end


function coco_utils.get_test(filename_)
  local filename = filename_ or '/storage/coco/test2015.txt'
  local file = io.open(filename, 'r')
  local filename_list ={}
  local url_list ={}
  local list_table = {}

  while true do
    local line = file:read()
    if not line then break end
    local item  =string.split(line, ',')
    local fname = item[1]
    local url = item[2]
    table.insert(filename_list, fname)
    table.insert(url_list, url)
    list_table[fname] = url
  end

  file:close()
  return filename_list, url_list, list_table
end

return coco_utils

