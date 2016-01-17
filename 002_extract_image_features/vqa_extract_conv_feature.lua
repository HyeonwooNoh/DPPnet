require 'torch'
require 'nn'
require 'image'

local load_caffe_model = require 'utils.load_caffe_model'
local vqa_loader = require 'utils.vqa_loader'
local qa_utils = require 'utils.qa_utils'
local stringx = require 'pl.stringx'

-- read input parameter
cmd = torch.CmdLine()
cmd:text()
cmd:text('extract convnet(VGG16) feature from vqa data')
cmd:text()
-- convnet model path
cmd:text('convnet model path')
cmd:option('-model_dir', 'model/VGG16_torch', 'dir containing model def *.lua *.prototxt, model param *.caffemodel')
cmd:option('-lua_model_name', 'VGG_ILSVRC_16_layers_extract_feature_proto.lua', 'model definition for torch')
cmd:option('-prototxt_name', 'VGG_ILSVRC_16_layers_deploy.prototxt', 'model definition for caffe')
cmd:option('-binary_name', 'VGG_ILSVRC_16_layers.caffemodel', 'model parameter for caffe')
-- gpu 
cmd:text('gpu usage')
cmd:option('-gpuid', 0, 'which GPU to use. -1 = use CPU')
-- data param
cmd:text('data param')
cmd:option('-coco_dir', './data/MSCOCO/images', 'directory contains coco images')
cmd:option('-vqa_data_dir', './data/VQA_torch', 'data directory. Should contain vqa data files')
cmd:option('-seq_len', 54, 'number of timesteps to unroll for')
cmd:option('-batch_size', 100, 'number of examples for each batch')
-- path for saving extracted features
cmd:text('path for saving extracted features')
cmd:option('-task_type', 'OpenEnded', 'options: [OpenEnded|MultipleChoice]')
cmd:option('-ans_type', 'major', 'options: [major|every]')
cmd:option('-save_dir', './data/vqa_vgg16_features', 'directory to save extracted features')
-- bookkeeping
cmd:option('-seed', 123, 'torch manual random number generator seed')

-- parse input params
opt = cmd:parse(arg or {})

-- convnet model path
local lua_model_name = paths.concat(opt.model_dir, opt.lua_model_name)
local prototxt_name = paths.concat(opt.model_dir, opt.prototxt_name)
local binary_name = paths.concat(opt.model_dir, opt.binary_name)

local save_feat_dir = opt.save_dir .. '/features'

-- initialize GPU
if opt.gpuid >= 0 then
   local ok, cunn = pcall(require, 'cunn')
   local ok2, cutorch = pcall(require, 'cutorch')
   if not ok then print('package cunn not found!') end
   if not ok2 then print('package cutorch not found!') end
   if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- load VGG16 convnet
local net = load_caffe_model.load(lua_model_name, prototxt_name, binary_name)
-- transfer network to gpu
if opt.gpuid >= 0 then
    net:cuda()
end
-- compute mean image
local mean_bgr = torch.Tensor({103.939, 116.779, 123.68})
local mean_bgr = mean_bgr:repeatTensor(224,224,1)
local mean_bgr = mean_bgr:permute(3,2,1)

-- read vqa data
local vqa_data = vqa_loader.load_data(opt.vqa_data_dir,opt.task_type,opt.ans_type, 
                                      opt.seq_len, opt.batch_size,true, 'all')

-- function for preprocess image for ConvNet
local function preprocess_img(localimg)
   local input_img = image.scale(localimg, 224,224)
   if input_img:size()[1] == 1 then
      input_img = input_img:repeatTensor(3,1,1)
   end
   input_img = input_img:index(1,torch.LongTensor{3,2,1})
   input_img = input_img * 255 - mean_bgr
   input_img = input_img:contiguous()
   return input_img
end  

-- extract feature for each ['train', 'val'] dataset
local function extract_feature(imageset, annotations)
   -- make sure that net is on evaluation mode
   net:evaluate()
   local img_ids = {}
   local cocopath = string.format('%s/%s', opt.coco_dir,imageset)
   print(string.format('start extracting feature for [%s]', imageset))
   for i = 1, #annotations do
      local ann = annotations[i]
      local image_id = ann.image_id
      if img_ids[image_id] == nil then
         local cocoimg_name = qa_utils.cocoimg_name(imageset, ann.image_id) 
         local img_path = paths.concat(cocopath, cocoimg_name)
         print(string.format('[%d/%d]: %s', i, #annotations, cocoimg_name))

         local img = image.load(img_path)
         local input = preprocess_img(img)
        
         if opt.gpuid >= 0 then
            input = input:float():cuda()
         end 

         local output = net:forward(input)
         local output_cpu = output:float()
     
         -- save extracted feature
         local cocofeat_name = stringx.split(cocoimg_name, '.')[1] .. '.t7'
         local save_path = paths.concat(save_feat_dir, cocofeat_name)
         torch.save(save_path, output_cpu)
         img_ids[image_id] = image_id
      end
      -- garbage collection
      if i % 100 == 0 then collectgarbage() end
   end
end 

-- create directory for saving extracted features
os.execute('mkdir -p ' .. save_feat_dir)

-- create log file
cmd:log(opt.save_dir .. '/log_feature_extraction', opt)

print(string.format('start extracting feature'))
print(string.format('save dir: %s', opt.save_dir))
-- extract features
print(string.format('start extracting train features'))
extract_feature('train2014', vqa_data.train_data.annotations)
print(string.format('start extracting validation features'))
extract_feature('val2014', vqa_data.val_data.annotations)
print(string.format('start extracting test features'))
extract_feature('test2015', vqa_data.test_data.question_list)
print(string.format('start extracting test-dev features'))
extract_feature('test2015', vqa_data.testdev_data.question_list)
print(string.format('done'))













