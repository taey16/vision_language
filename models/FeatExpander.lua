
require 'nn'
require 'cunn'
require 'cudnn'


-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')


function layer:__init(n)
  parent.__init(self)
  self.n = n
end


function layer:updateOutput(input)
  -- act as a noop for efficiency
  if self.n == 1 then 
    self.output = input; 
    return self.output 
  end

  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2)
  local d = input:size(2)
  self.output:resize(input:size(1)*self.n, d)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.output[{ {j,j+self.n-1} }] = 
      input[{ {k,k}, {} }]:expand(self.n, d) -- copy over
  end
  return self.output
end


function layer:updateGradInput(input, gradOutput)
  -- act as noop for efficiency
  if self.n == 1 then 
    self.gradInput = gradOutput; 
    return self.gradInput 
  end

  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end


