local HashLinear, parent = torch.class('nn.HashLinear', 'nn.Linear')

function HashLinear:__init(inputSize, outputSize, config)
    parent.__init(self, inputSize, outputSize)

    self:copyConfig(config)

    ------ Size Info
    self.size_in    = inputSize
    self.size_out   = outputSize

    self.size_w     = inputSize*outputSize
    self.size_b     = outputSize

    self.hsize_w    = config.hsize_w

    if self.config.hbias then 
        self.hsize_b = config.hsize_b
    else
        self.hsize_b = self.size_b
    end

    -- Hashed Parameter & Gradient Preallocation
    self.h_weight           = torch.Tensor(self.hsize_w):zero()
    self.h_gradWeight       = torch.Tensor(self.hsize_w):zero()
    self.h_bias             = torch.Tensor(self.hsize_b):zero()
    self.h_gradBias         = torch.Tensor(self.hsize_b):zero()

    self.xxhash = require 'xxhash'

    self:HashConfig('W')
    if self.config.hbias then
        self:HashConfig('B')
    end
    self:hashReset()
end

function HashLinear:hashReset(stdv)
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1./math.sqrt(self.size_in)
    end
    self.h_weight:uniform(-stdv, stdv)
    self.h_bias:uniform(-stdv, stdv)
end

function HashLinear:hashFunc(hN, size_out, size_in, extra_str)
    local Nall = size_out * size_in
    local idx = torch.FloatTensor(Nall,1)
    local max = hN
    local min = 1
    local rep = 3
    local range = max - min + 1
    local count = 0
    local extra_str = extra_str or ''
    local key_i, key_j
    for i = 1, size_out do
        for j = 1, size_in do
            count, key_i, key_j = count+1, '', '' 
            for r = 0,rep-1 do
                key_i = key_i .. tostring(i+r)
                key_j = key_j .. tostring(j+r)
            end
            key = key_i .. '_' .. key_j .. extra_str

            idx[count] = self.xxhash.xxh32(key,self.config.hashSeed) % (range) + min
        end
    end
    return idx:cuda()
end


function HashLinear:copyConfig(config)
    if type(config) ~= 'table' then
        error('The third argument \"config\" should be a table')
    end
    self.config              = {}
    self.config.hsize_w      = config.hsize_w
    self.config.hsize_b      = config.hsize_b
    self.config.xi           = config.xi
    self.config.hbias        = config.hbias
    self.config.hashSeed     = config.hashSeed or 1691
    self.config.rescale_grad = config.rescale_grad
    self.config.verbose      = config.verbose
    self.config.cpu          = config.cpu or false
    if not self.config.hsize_w then
        error('variable hsize_w must be specified')
    end
    if self.config.cpu then error('only cuda is supported now!') end
    if self.config.verbose then
        if self.config.xi then print('Using xi auxiliary hash function') end
        if self.config.hbias then print('Hashing bias terms') end
    end
end

-- WorB is either W or B
function HashLinear:HashConfig(WorB)
    local h_size, dim1, dim2
    if WorB == 'W' then
        h_size = self.hsize_w
        dim1 = self.size_out
        dim2 = self.size_in
    elseif WorB == 'B' then
        h_size = self.hsize_b
        dim1 = self.size_out
        dim2 = 1
    end
    self['idx' .. WorB] = self:hashFunc(h_size, dim1, dim2, 'idx' ..WorB) :reshape(dim1,dim2)
    if self.config.xi then
        self['xi_' .. WorB] = self:hashFunc(2, dim1, dim2, 'xi_' .. WorB)
                      :reshape(dim1,dim2):mul(2):add(-3) -- convert to 1 or -1
    end

    -- Important! should be initialized with 0
    self['unique_idx' .. WorB] = torch.zeros(h_size,1,'torch.CudaTensor')
    self['sort_val_' .. WorB] = torch.range(1,dim1*dim2,'torch.FloatTensor'):reshape(dim1,dim2):cuda()
    self['sort_key_' .. WorB] = self['idx' .. WorB]:clone()
    libhashnn.mysort(self['sort_key_' .. WorB],self['sort_val_'.. WorB])


    ------ Compute occupancy for w
    self['occupancy_' .. WorB] = self['unique_idx' .. WorB]:clone()
    self['buffer_' .. WorB]    = self['occupancy_' .. WorB]:clone()

    local allones = self['sort_key_' .. WorB]:clone():fill(1)
    libhashnn.myreduce(self['sort_key_' .. WorB], allones, self['unique_idx' .. WorB], self['occupancy_' .. WorB], self['buffer_' .. WorB])

end



function HashLinear:updateOutput(input)
  -- Reconstruct weight matrix and bias vector
  libhashnn.myindexing(self.h_weight, self.idxW, self.weight);
  if self.config.hbias then
    libhashnn.myindexing(self.h_bias, self.idxB, self.bias);
  else 
    self.bias:copy(self.h_bias)
  end
  if self.config.xi then
    self.weight:cmul(self.xi_W)
    if self.config.hbias then
      self.bias:cmul(self.xi_B)
    end
  end

  self.output = parent.updateOutput(self, input)
  return self.output
end



function HashLinear:accGradParameters(input, gradOutput, scale)
    -- Reset gradients
    self.gradWeight:zero()
    self.gradBias:zero()

    parent.accGradParameters(self, input, gradOutput, scale)

    --accumarray part
    if not self.gradWBuffer or not self.gradBBuffer then
        self.gradWBuffer = self.gradWeight:clone()
        if self.config.hbias then self.gradBBuffer = self.gradBias:clone()
        else self.gradBBuffer = "place_holder" end
    end

    if not self.idxW_buffer or not self.idxB_buffer then
        self.idxW_buffer = self.idxW:clone()
        if self.config.hbias then self.idxB_buffer = self.idxB:clone()
        else self.idxB_buffer = "place_holder" end
    end

    if self.config.xi then
        self.gradWeight:cmul(self.xi_W)
        if self.config.hbias then
            self.gradBias:cmul(self.xi_B)
        end
    end

    libhashnn.myindexing(self.gradWeight, self.sort_val_W, self.gradWBuffer)
    libhashnn.myreduce(self.sort_key_W,self.gradWBuffer,self.unique_idxW,self.h_gradWeight,self.buffer_W)
    if self.config.rescale_grad then
      self.h_gradWeight:cdiv(self.occupancy_W)
    end

    if self.config.hbias then
        libhashnn.myindexing(self.gradBias, self.sort_val_B, self.gradBBuffer)
        libhashnn.myreduce(self.sort_key_B,self.gradBBuffer,self.unique_idxB,self.h_gradBias,self.buffer_B)
        if self.config.rescale_grad then
          self.h_gradBias:cdiv(self.occupancy_B)
        end
    else
        self.h_gradBias:copy(self.gradBias)
    end

end





function HashLinear:parameters()
    return {self.h_weight, self.h_bias}, {self.h_gradWeight, self.h_gradBias}
end

