local Linear_wo_bias, parent = torch.class('nn.Linear_wo_bias', 'nn.Module')

function Linear_wo_bias:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)

   self:reset()
end

function Linear_wo_bias:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end

   return self
end

function Linear_wo_bias:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      self.output:zero()
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
         self.addBuffer = input.new(nframe):fill(1)
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function Linear_wo_bias:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end

      return self.gradInput
   end
end

function Linear_wo_bias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   end
end

-- we do not need to accumulate parameters when sharing
Linear_wo_bias.sharedAccUpdateGradParameters = Linear_wo_bias.accUpdateGradParameters


function Linear_wo_bias:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
