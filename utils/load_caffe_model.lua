
-- this is additional utils for using 'loadcaffe' module
-- that is 'loadcaffe' module have to be installed

local load_caffe_model = {}

require 'loadcaffe'
local ffi = require 'ffi'
local C = loadcaffe.C

function load_caffe_model.load(lua_model_name, prototxt_name, binary_name)
   local handle = ffi.new('void*[1]')
     
   -- loads caffe model in memory and keeps handle to it in ffi
   local old_val = handle[1]
   print(string.format('load binary weight'))
   C.loadBinary(handle, prototxt_name, binary_name)
   if old_val == handle[1] then return end

   -- executes the script, defining global 'model' module list
   print(string.format('load lua prototxt file..'))
   if lua_model_name == nil or paths.filep(lua_model_name) == false then
      print(string.format('wrong lua_model_name'))
   end
   local model = dofile(lua_model_name)
   print(string.format('done'))

   -- goes over the list, copying weights from caffe blobs to torch tensor
   print(string.format('start copying weight from caffe blobs to torch tensor'))
   local net = nn.Sequential()
   local list_modules = model
   for i,item in ipairs(list_modules) do
      print(string.format('loading... [%d]: %s', i, item[1]))
      if item[2].weight then
         local w = torch.FloatTensor()
         local bias = torch.FloatTensor()
         C.loadModule(handle, item[1], w:cdata(), bias:cdata())
         if backend == 'ccn2' then
            w = w:permute(2,3,4,1)
         end
         item[2].weight:copy(w)
         item[2].bias:copy(bias)
       end
       net:add(item[2])
   end
   print(string.format('done'))
   C.destroyBinary(handle)

   return net
end

return load_caffe_model
