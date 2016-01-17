-- Memory Efficient Hasher (process only one batch)
HasherME = {}
setmetatable(HasherME, HasherME)

function HasherME:init(inputSize, outputSize, config)

   self:copyConfig(config)
    ------ Size Info
    self.size_in    = inputSize
    self.size_out   = outputSize

    self.size_w     = inputSize*outputSize
    self.hsize_w    = config.hsize_w
 
    self.xxhash = require 'xxhash'
    self:HashConfig('W')
end
function HasherME:hashFunc(hN, size_out, size_in, extra_str)
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
function HasherME:copyConfig(config)
    if type(config) ~= 'table' then
        error('The third argument \"config\" should be a table')
    end
    self.config              = {}
    self.config.hsize_w      = config.hsize_w
    self.config.xi           = config.xi
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
    end
end
-- WorB is either W or B
function HasherME:HashConfig(WorB)
    local h_size, dim1, dim2
    if WorB == 'W' then
        h_size = self.hsize_w
        dim1 = self.size_out
        dim2 = self.size_in
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

function HasherME:forward(input)
    if not self.output then
        self.output = torch.zeros(self.size_in, self.size_out):cuda()
    end
    libhashnn.myindexing(input, self.idxW, self.output)
    if self.config.xi then
        self.output:cmul(self.xi_W)
    end
    return self.output
end

function HasherME:backward(gradOutput)
    if not self.gradInput then 
        self.gradInput = torch.zeros(self.hsize_w):cuda()
    end
    if not self.gradOBuffer then
        self.gradOBuffer = gradOutput:clone()
    end

    if self.config.xi then
        gradOutput:cmul(self.xi_W)
    end
    libhashnn.myindexing(gradOutput, self.sort_val_W, self.gradOBuffer)
    libhashnn.myreduce(self.sort_key_W,self.gradOBuffer,self.unique_idxW,self.gradInput,self.buffer_W)

    if self.config.rescale_grad then
        self.gradInput:cdiv(self.occupancy_W)
    end

    return self.gradInput
end


return 0

