require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'model.layers.Linear_wo_bias'

local skipthoughts_GRU = require 'model.skipthoughts_GRU'
local load_caffe_model = require 'utils.load_caffe_model'
local vqa_loader = require 'utils.vqa_loader'
local model_utils = require 'utils.model_utils'
local cjson = require 'cjson'

-- read input parameter
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train DPPnet from CNN_FIXED model')
cmd:text()
cmd:text('Options')
-- general
cmd:option('-init_from', './model/DPPnet/CNN_FIXED_vqa.t7', 
           'initialize network parameters from checkpoint at this path')
cmd:option('-continue_learning', 'false', 'when init_from is defined, you could recover same optimization state by setting this value')
cmd:option('-loss_explod_threshold', 10, 'loss explosion threshold')
-- data
cmd:option('-vqa_data_dir', './data/VQA_torch', 'data directory. Should contain vqa data files')
cmd:option('-cache_dir', './cache', 'directory containing caches')
cmd:option('-feat_dir', './data/vqa_vgg16_features/features', 'vqa feature directory')
cmd:option('-cocoimg_dir', './data/MSCOCO/images', 'vqa feature directory')
cmd:option('-vqa_task_type', 'OpenEnded', 'task type: [OpenEnded|MultipleChoice]')
cmd:option('-vqa_ans_type', 'major', 'answer type: [major|every]')
cmd:option('-testset', 'test-dev2015', 'option: [test-dev2015|test2015]')
cmd:option('-alg_name', 'DPPnet_vqa', 'algorithm name for vqa eval result file')
-- model params
cmd:option('-conv_feat_dim', 4096, 'dimension of extracted convnet feature')
cmd:option('-hash_size_w', 40000, 'hash size')
-- cnn params
cmd:option('-cnn_model_dir', 'model/VGG16_torch', 'dir containing model def *.lua *.prototxt, model param *.caffemodel')
cmd:option('-cnn_lua_model_name', 'VGG_ILSVRC_16_layers_extract_feature_proto.lua', 'model definition for torch')
cmd:option('-cnn_prototxt_name', 'VGG_ILSVRC_16_layers_deploy.prototxt', 'model definition for caffe')
cmd:option('-cnn_binary_name', 'VGG_ILSVRC_16_layers.caffemodel', 'model parameter for caffe')
cmd:option('-cnn_weight_decay', 0.0005, 'weight decay parameter for cnn')
-- rnn params
cmd:option('-path_uni_gru_params', './data/skipthoughts_params/uni_gru_params.t7',
           'path for uni-gru parameters for skipthoughts vectors')
cmd:option('-path_uni_gru_word2vec', './data/skipthoughts_params/vqa_trainval_uni_gru_word2vec.t7',
           'path for uni-gru word2vec parameters')
cmd:option('-start_finetune_after', 0, 
           'start finetuning after n epoch [0] if you want to finetune from the first time')
cmd:option('-stop_finetune_after', 200, 
           'stop finetuning after n epoch [200] if you dont want to stop finetune')
-- optimization
cmd:option('-optimizer', 'adam', 'option: adam')
cmd:option('-batch_order_option', 1, '[1]:shuffle, [2]:inorder, [3]:sort, [4]:randsort')
cmd:option('-max_epochs',6,'number of full passes through the training data')
cmd:option('-learning_rate',0.00001,'learning rate')
cmd:option('-learning_rate_decay',0.8,'learning rate decay, if you dont want to decay learning rate, set 1')
cmd:option('-learning_rate_decay_after',3,'in number of epochs, when to start decaying the learning rate')
cmd:option('-seq_len', 54, 'number of timesteps to unroll for')
cmd:option('-batch_size', 32, 'number of examples for each batch')
cmd:option('-test_batch_size', 32, 'number of examples for each test batch')
cmd:option('-grad_clip', 0.1, 'clip gradients at this value [almost no gradient clipping is applied')
-- save
cmd:option('-save_dir', 'save_result_vqa', 'subdirectory to save results [log, snapshot]')
cmd:option('-log_dir', 'training_log', 'subdirectory to log experiments in')
cmd:option('-snapshot_dir', 'snapshot', 'subdirectory to save checkpoints')
cmd:option('-results_dir', 'results', 'subdirectory for saving test results')
-- bookkeeping
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000000,'every how many iterations should we evaluate on validation data?')
cmd:option('-seed', 123, 'torch manual random number generator seed')
-- GPU/CPU
cmd:option('-gpuid', 0, 'which GPU to use. -1 = use CPU')

-- parse input params
opt = cmd:parse(arg or {})

torch.manualSeed(opt.seed)

-- create directory and log file
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.log_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.snapshot_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.results_dir))
cmd:log(string.format('%s/%s/log_cmdline', opt.save_dir, opt.log_dir), opt)
print('create directory and log file')

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
        LookupTable = nn.LookupTable
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- require HashedNets module
require 'HasherME'
require 'libhashnn'

-- load vqa dataset
print('load vqa dataset')
local vqa_data = vqa_loader.load_data(opt.vqa_data_dir, opt.vqa_task_type, opt.vqa_ans_type,
                                      opt.seq_len, opt.batch_size, true, opt.testset, opt.test_batch_size,
                                      false, opt.cache_dir)

-- add <eos>
vqa_data.vocab_size = vqa_data.vocab_size + 1
vqa_data.vocab_dict[vqa_data.vocab_size] = '<eos>'
vqa_data.vocab_map['<eos>'] = vqa_data.vocab_size

print(string.format('num data train: %d', vqa_data.train_data.ex_num_train))
print(string.format('num data test: %d', vqa_data.test_data.ex_num_train))

-- load uni gru param and word vectors
print('load uni gru param and word vectors')
local uparams = torch.load(opt.path_uni_gru_params)
local utables = torch.load(opt.path_uni_gru_word2vec)
opt.rnn_size = uparams.Ux:size(1)
opt.word_dim = utables:size(2)

-- cnn model params
local lua_model_name = paths.concat(opt.cnn_model_dir, opt.cnn_lua_model_name)
local prototxt_name = paths.concat(opt.cnn_model_dir, opt.cnn_prototxt_name)
local binary_name = paths.concat(opt.cnn_model_dir, opt.cnn_binary_name)

-- currently, only random initialization is implemented
-- which means, every networks should be trained from scratch
local protos
local checkpoint_start_train_iter
local do_random_init = true
local batch_norm_layer
local hash_config = {}
if string.len(opt.init_from) > 0 then
   print ('loading a neural network from checkpoint ' .. opt.init_from)
   local checkpoint = torch.load(opt.init_from)
   protos = checkpoint.protos
   checkpoint_start_train_iter = checkpoint.i
   -- make sure vocabs are the same
   local vocab_compatible = true
   for c, i in pairs(checkpoint.vocab_map) do
      if not vqa_data.vocab_map[c] == i then
         vocab_compatible = false
         break
      end
   end
   assert(vocab_compatible, 'error, question vocabulary is incompatible')
   for c, i in pairs(checkpoint.answer_map) do
      if not vqa_data.answer_map[c] == i then
         vocab_compatible = false
         break
      end
   end
   assert(vocab_compatible, 'error, answer vocabulary is incompatible')
   if not vqa_data.vocab_size == checkpoint.vocab_size then
      vocab_compatible = false
   end 
   assert(vocab_compatible, 'error, vocabulary size is incompatible')
   if not vqa_data.answer_size == checkpoint.answer_size then
      vocab_compatible = false
   end 
   assert(vocab_compatible, 'error, answer size is incompatible')
   local feat_dim_compatible = true
   if not opt.conv_feat_dim == checkpoint.conv_feat_dim then
      feat_dim_compatible = false
   end
   assert(feat_dim_compatible, 'error, convnet feature dimension is incompatible')
   print('overwriting neural network parameter setting')
   opt.word_dim = checkpoint.word_dim
   print(string.format('word_dim : %d', opt.word_dim))
   if opt.continue_learning == 'true' then
      print('overwriting optimization state')
      opt.optimizer = checkpoint.optimizer
      print(string.format('optimizer: %s', opt.optimizer))
      opt.optim_state = checkpoint.optim_state
   end
   do_random_init = false

   -- remove dropout layer
   local dropout_idx
   for k, v in pairs(protos.net.modules) do
      if v.__typename == 'nn.Dropout' then
         dropout_idx = k
      end
   end
   if dropout_idx ~= nil then
      protos.net:remove(dropout_idx)
   end
   -- make pointer to batch normalization layer
   for k, v in pairs(protos.multimodal.modules) do
      if v.__typename == 'nn.BatchNormalizaiton' then
         batch_norm_layer = v
      end
   end

   hash_config = checkpoint.hash_config
   HasherME:init(2000,2000,hash_config)
else
   assert(false, 'pre-trained model is required to run this script')
end
-- loading pretrained cnn
print('loading pretrained cnn')
protos.cnn = load_caffe_model.load(lua_model_name, prototxt_name, binary_name)

-- transfer to gpu
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end


-- combine all parameters [for optimization]
print('combine all parameters [for optimization]')
params, grad_params = model_utils.combine_all_parameters(protos.cnn,
                                                         protos.net, 
                                                         protos.rnn, protos.word_embed, protos.linear, 
                                                         protos.bias_term,
                                                         protos.multimodal) 

-- clone many times for rnn sequential reading
local mult_protos = {}
print('clone many times for rnn sequential reading')
mult_protos.rnns = model_utils.clone_many_times(protos.rnn, opt.seq_len)
mult_protos.word_embeds = model_utils.clone_many_times(protos.word_embed, opt.seq_len)

local cnn_param, cnn_grad = protos.cnn:parameters()
local rnn_param, rnn_grad = protos.rnn:parameters()
local embed_param, embed_grad = protos.word_embed:parameters()
local cls_params, cls_grad = protos.net:parameters()
local linear_params, linear_grad = protos.linear:parameters()
local multimodal_params, multimodal_grad = protos.multimodal:parameters()
local bias_params, bias_grad = protos.bias_term:parameters()

local function compute_norm_grad(layer_grad)
   local norm_grad = 0
   for k, v in pairs(layer_grad) do
      local norm_loc = v:norm()
      norm_grad = norm_grad + norm_loc * norm_loc
   end
   return torch.sqrt(norm_grad)
end 
local function compute_norm_param_grad(layer_params, layer_grad) 
   local norm_grad = 0
   local norm_param = 0
   for k, v in pairs(layer_params) do
      local norm_loc = v:norm()
      norm_param = norm_param + norm_loc * norm_loc
   end
   for k, v in pairs(layer_grad) do
      local norm_loc = v:norm()
      norm_grad = norm_grad + norm_loc * norm_loc
   end
   return torch.sqrt(norm_param), torch.sqrt(norm_grad)
end

-- gru init state
local init_state = torch.zeros(opt.batch_size, opt.rnn_size)
if opt.gpuid >= 0 then
   init_state = init_state:float():cuda()
end
local rnn_out = torch.Tensor(opt.batch_size, opt.rnn_size)
if opt.gpuid >= 0 then
   rnn_out = rnn_out:float():cuda()
end
local drnn_out_t = torch.Tensor(opt.batch_size, opt.rnn_size)
if opt.gpuid >= 0 then
   drnn_out_t = drnn_out_t:float():cuda()
end
local bias_input = torch.zeros(opt.batch_size) + 1
if opt.gpuid >= 0 then
   bias_input = bias_input:float():cuda()
end
local gate_buffer = torch.Tensor(opt.batch_size, 2000)
if opt.gpuid >= 0 then
   gate_buffer = gate_buffer:float():cuda()
end
local dgate_buffer = torch.Tensor(opt.batch_size, opt.hash_size_w)
if opt.gpuid >= 0 then
   dgate_buffer = dgate_buffer:float():cuda()
end
local dnet_buffer = torch.Tensor(opt.batch_size, 2000)
if opt.gpuid >= 0 then
   dnet_buffer = dnet_buffer:float():cuda() 
end
-- gru init state
local test_init_state = torch.zeros(opt.test_batch_size, opt.rnn_size)
if opt.gpuid >= 0 then
   test_init_state = test_init_state:float():cuda()
end
local test_rnn_out = torch.Tensor(opt.test_batch_size, opt.rnn_size)
if opt.gpuid >= 0 then
   test_rnn_out = test_rnn_out:float():cuda()
end
local test_bias_input = torch.zeros(opt.test_batch_size) + 1
if opt.gpuid >= 0 then
   test_bias_input = test_bias_input:float():cuda()
end
local test_gate_buffer = torch.Tensor(opt.test_batch_size, 2000)
if opt.gpuid >= 0 then
   test_gate_buffer = test_gate_buffer:float():cuda()
end
apply_finetuning = false
if opt.start_finetune_after == 0 and stop_finetune_after ~= 0 then
   apply_finetuning = true
end

train_acc = 0
train_ex_num = 0
-- function for optimization
function feval(x)
   if x ~= params then
      params:copy(x)
   end
   grad_params:zero()

   ------------------- get minibatch -------------------
   local imgs, x, x_len, y = vqa_data.train_data:next_batch_image(vqa_data.vocab_map,
                                                       vqa_data.answer_map,
                                                       opt.cocoimg_dir, 224, 224)
   if opt.gpuid >= 0 then
      imgs = imgs:float():cuda()
      x = x:float():cuda()
      y = y:float():cuda()
   end
   -- make sure we are in correct omde (this is cheap, sets flag)
   protos.cnn:training()
   protos.net:training()
   protos.rnn_dropout:training()
   protos.linear:training()
   protos.multimodal:training()
   protos.bias_term:training()

   -------------------- forward pass -------------------
   local loss = 0
   -- cnn forward
   local cnn_feats = protos.cnn:forward(imgs)
   print (cnn_feats:max())
   -- net forward
   local net_out = protos.net:forward(cnn_feats)
   -- rnn forward
   local max_len = x_len:max()
   local min_len = x_len:min()
   local we_vecs = {}
   local rnn_state = {[0] = init_state}
   rnn_out:zero()
   for t = 1, max_len do
      mult_protos.rnns[t]:training()
      mult_protos.word_embeds[t]:training()
      local we = mult_protos.word_embeds[t]:forward(x[{t,{}}])
      local lst = mult_protos.rnns[t]:forward({we, rnn_state[t-1]})
      we_vecs[t] = we
      rnn_state[t] = lst
      if t >= min_len then
         for k = 1, opt.batch_size do
            if x_len[k] == t then
               rnn_out[k] = lst[k]
            end
         end
      end
   end
   -- dropout forward
   local dropout_out = protos.rnn_dropout:forward(rnn_out)
   -- linear forward
   local linear_out = protos.linear:forward(dropout_out)
   -- hashed gating
   local gated = gate_buffer
   for i = 1, opt.batch_size do
      local hashed_out = HasherME:forward(linear_out[i])
      local rep_net_out = torch.reshape(net_out[i], net_out:size(2), 1)
      gated[i] = torch.squeeze(protos.MM:forward({hashed_out, rep_net_out}))
   end
   -- compute bias
   local bias = protos.bias_term:forward(bias_input)
   -- multimodal classifier
   local multimodal_out = protos.multimodal:forward({gated,bias})
   -- logsoftmax
   local logsoftmax = protos.logsoftmax:forward(multimodal_out)
   -- compute accuracy
   local max_score, ans = torch.max(logsoftmax, 2)
   ans = torch.squeeze(ans)
   train_acc = train_acc + torch.eq(ans, y):sum()
   train_ex_num = train_ex_num + y:nElement()
   -- loss forward
   loss = loss + protos.criterion:forward(logsoftmax, y)

   -------------------- backward pass -------------------
   -- loss backward
   local dcriterion = protos.criterion:backward(logsoftmax, y)
   local dlogsoftmax = protos.logsoftmax:backward(multimodal_out, dcriterion)
   local dmultimodal = protos.multimodal:backward({gated,bias}, dlogsoftmax)
   local dgated = dmultimodal[1]
   local dbias = dmultimodal[2]
   -- bias backward
   local dbias_input = protos.bias_term:backward(bias_input, dbias)
   -- hashed gating backward
   local dlinear_out = dgate_buffer
   local dnet_out = dnet_buffer
   for i = 1, opt.batch_size do
      local rep_dgated = torch.reshape(dgated[i], dgated:size(2), 1)
      local hashed_out = HasherME:forward(linear_out[i])
      local rep_net_out = torch.reshape(net_out[i], net_out:size(2), 1)
      local dMM = protos.MM:backward({hashed_out, rep_net_out}, rep_dgated)
      local dhashed_out = dMM[1]
      dlinear_out[i] = HasherME:backward(dhashed_out)
      dnet_out[i] = torch.squeeze(dMM[2])
   end
   -- rnn linear backward
   local ddropout_out = protos.linear:backward(dropout_out, dlinear_out)
   local drnn_out = protos.rnn_dropout:backward(rnn_out, ddropout_out)
   -- rnn backward
   if apply_finetuning then 
      local drnn_state = {[max_len+1] = init_state:clone()}
                 -- true also zeros the clones
      for t = max_len, 1, -1 do
         drnn_out_t:copy(drnn_state[t+1])
         if t >= min_len then
            for k = 1, opt.batch_size do
               if x_len[k] == t then
                  drnn_out_t[k] = drnn_out[k]
               end
            end
         end
         local dlst = mult_protos.rnns[t]:backward({we_vecs[t], rnn_state[t-1]}, drnn_out_t)

         drnn_state[t] = dlst[2] -- dlst[1] is gradient on x, which we don't need
      end
   end
   -- net backward
   local dcnn_feats = protos.net:backward(cnn_feats, dnet_out)
   -- cnn backward
   local dimgs = protos.cnn:backward(imgs, dcnn_feats)

   ------------------ CNN weight decaying ---------------
   if opt.cnn_weight_decay ~= 0 then
      for k, v in pairs(cnn_grad) do
         v:add(opt.cnn_weight_decay, cnn_param[k])
      end
      print (string.format(' - cnn weight decay: %f', opt.cnn_weight_decay))
   end

   ------------------ gradient clipping ---------------
   local grad_norm
   grad_norm = compute_norm_grad(cnn_grad)
   if grad_norm > opt.grad_clip then
      for k, v in pairs(cnn_grad) do
         v:mul(opt.grad_clip / grad_norm)
      end
      print (string.format(' - cnn grad clipped norm: [%f -> %f]', grad_norm, opt.grad_clip))
   else
      print (string.format(' - cnn grad is not clipped norm: %f', grad_norm))
   end
   grad_norm = compute_norm_grad(cls_grad)
   if grad_norm > opt.grad_clip then
      for k, v in pairs(cls_grad) do
         v:mul(opt.grad_clip / grad_norm)
      end
      print (string.format(' - image classifier grad clipped norm: [%f -> %f]', grad_norm, opt.grad_clip))
   else
      print (string.format(' - image classifier grad is not clipped norm: %f', grad_norm))
   end
   if apply_finetuning then 
      grad_norm = compute_norm_grad(rnn_grad)
      if grad_norm > opt.grad_clip then
         for k, v in pairs(rnn_grad) do
            v:mul(opt.grad_clip / grad_norm)
         end
         print (string.format(' - rnn grad clipped norm: [%f -> %f]', grad_norm, opt.grad_clip))
      else
         print (string.format(' - rnn grad is not clipped norm: %f', grad_norm))
      end
   end
   grad_norm = compute_norm_grad(linear_grad)
   if grad_norm > opt.grad_clip then
      for k, v in pairs(linear_grad) do
         v:mul(opt.grad_clip / grad_norm)
      end
      print (string.format(' - linear grad clipped norm: [%f -> %f]', grad_norm, opt.grad_clip))
   else
      print (string.format(' - linear grad is not clipped norm: %f', grad_norm))
   end
   grad_norm = compute_norm_grad(multimodal_grad)
   if grad_norm > opt.grad_clip then
      for k, v in pairs(multimodal_grad) do
         v:mul(opt.grad_clip / grad_norm)
      end
      print (string.format(' - multimodal grad clipped norm: [%f -> %f]', grad_norm, opt.grad_clip))
   else
      print (string.format(' - multimodal grad is not clipped norm: %f', grad_norm))
   end
   grad_norm = compute_norm_grad(bias_grad)
   if grad_norm > opt.grad_clip then
      for k, v in pairs(bias_grad) do
         v:mul(opt.grad_clip / grad_norm)
      end
      print (string.format(' - bias grad clipped norm: [%f -> %f]', grad_norm, opt.grad_clip))
   else
      print (string.format(' - bias grad is not clipped norm: %f', grad_norm))
   end
   return loss, grad_params
end

function predict_result (imgs, x, x_len)
   -- transfer to gpu
   if opt.gpuid >= 0 then
      imgs = imgs:float():cuda()
      x = x:float():cuda()
   end
   -- make sure we are in correct omde (this is cheap, sets flag)
   protos.cnn:evaluate()
   protos.net:evaluate()
   protos.rnn_dropout:evaluate()
   protos.linear:evaluate()
   protos.multimodal:evaluate()
   protos.bias_term:evaluate()
   -------------------- forward pass -------------------
   local loss = 0
   -- cnn forward
   local cnn_feats = protos.cnn:forward(imgs)
   -- net forward
   local net_out = protos.net:forward(cnn_feats)
   -- rnn forward
   local max_len = x_len:max()
   local min_len = x_len:min()
   local we_vecs = {}
   local rnn_state = {[0] = test_init_state}
   test_rnn_out:zero()
   for t = 1, max_len do
      mult_protos.rnns[t]:evaluate()
      mult_protos.word_embeds[t]:evaluate()
      local we = mult_protos.word_embeds[t]:forward(x[{t,{}}])
      local lst = mult_protos.rnns[t]:forward({we, rnn_state[t-1]})
      we_vecs[t] = we
      rnn_state[t] = lst
      if t >= min_len then
         for k = 1, opt.test_batch_size do
            if x_len[k] == t then
               test_rnn_out[k] = lst[k]
            end
         end
      end
   end
   -- dropout forward
   local dropout_out = protos.rnn_dropout:forward(test_rnn_out)
   -- linear forward
   local linear_out = protos.linear:forward(dropout_out)
   -- hashed gating
   local gated = test_gate_buffer
   for i = 1, opt.test_batch_size do
      local hashed_out = HasherME:forward(linear_out[i])
      local rep_net_out = torch.reshape(net_out[i], net_out:size(2), 1)
      gated[i] = torch.squeeze(protos.MM:forward({hashed_out, rep_net_out}))
   end
   -- compute bias
   local bias = protos.bias_term:forward(test_bias_input)
   -- multimodal classifier
   local multimodal_out = protos.multimodal:forward({gated,bias})
   -- logsoftmax
   local logsoftmax = protos.logsoftmax:forward(multimodal_out)

   return logsoftmax
end

---------------------------- training -----------------------------

-- create log file for optimization
trainLogger = optim.Logger(paths.concat(string.format('%s/%s',opt.save_dir,opt.log_dir), 'train.log'))
testLogger = optim.Logger(paths.concat(string.format('%s/%s',opt.save_dir,opt.log_dir), 'test.log'))

-- align train / val data
vqa_data.train_data:set_batch_order_option(opt.batch_order_option)

vqa_data.train_data:reorder()

local optim_state
local optimizer
if  opt.optimizer == 'adam' then -- adam
   if string.len(opt.init_from) > 0 and opt.continue_learning == 'true' then
      optim_state = opt.optim_state
   else
      optim_state = {learningRate = opt.learning_rate}
   end
   optimizer = optim.adam
else
   assert(true, 'wrong optimizer option - not yet implemented')
end

local iterations_per_epoch = vqa_data.train_data.iter_per_epoch
local iterations = opt.max_epochs * iterations_per_epoch 
local loss0 = nil
local start_train_iter = 1
-- continue training
if string.len(opt.init_from) > 0 then
   if opt.continue_learning == 'true' then
      start_train_iter = checkpoint_start_train_iter
   end
end
for i = start_train_iter, iterations do
   local epoch = i / iterations_per_epoch
   print('-----------------------------------------------------------------------')
   local timer = torch.Timer()
   local _, loss = optimizer(feval, params, optim_state)
   local time = timer:time().real
   
   local train_loss = loss[1] -- the loss is inside a list, pop it
   if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), lr=%f, loss=%6.8f, gnorm=%6.4e, pnorm=%6.4e, time=%.2fs",
      i, iterations, epoch,optim_state.learningRate, train_loss, grad_params:norm(), params:norm(), time))

      if apply_finetuning then 
         local cnn_norm_param, cnn_norm_grad = compute_norm_param_grad(cnn_param, cnn_grad)
         local rnn_norm_param, rnn_norm_grad = compute_norm_param_grad(rnn_param, rnn_grad)
         local cls_norm_param, cls_norm_grad = compute_norm_param_grad(cls_params, cls_grad)
         local linear_norm_param, linear_norm_grad = compute_norm_param_grad(linear_params, linear_grad)
         local multimodal_norm_param, multimodal_norm_grad = compute_norm_param_grad(multimodal_params, multimodal_grad)
         local bias_norm_param, bias_norm_grad = compute_norm_param_grad(bias_params, bias_grad)
      
         print(string.format(" [norm] VP:%6.4e, RP:%6.4e, CP:%6.4e, LP:%6.4e, MP:%6.4e, BP:%6.4e", 
                            cnn_norm_param, rnn_norm_param, cls_norm_param, linear_norm_param, multimodal_norm_param,bias_norm_param))
         print(string.format(" [grad] VG:%6.4e, RG:%6.4e, CG:%6.4e, LG:%6.4e, MG:%6.4e, BG:%6.4e", 
                            cnn_norm_grad, rnn_norm_grad, cls_norm_grad, linear_norm_grad, multimodal_norm_grad,bias_norm_grad))
      else
         local cnn_norm_param, cnn_norm_grad = compute_norm_param_grad(cnn_param, cnn_grad)
         local cls_norm_param, cls_norm_grad = compute_norm_param_grad(cls_params, cls_grad)
         local linear_norm_param, linear_norm_grad = compute_norm_param_grad(linear_params, linear_grad)
         local multimodal_norm_param, multimodal_norm_grad = compute_norm_param_grad(multimodal_params, multimodal_grad)
         local bias_norm_param, bias_norm_grad = compute_norm_param_grad(bias_params, bias_grad)
         print(string.format(" [norm] VP:%6.4e, CP:%6.4e, LP:%6.4e [grad] VG:%6.4e, CG:%6.4e, LG:%6.4e", 
                               cnn_norm_param, cls_norm_param, linear_norm_param,
                               cnn_norm_grad, cls_norm_grad, linear_norm_grad))
         print(string.format(" [norm] MP:%6.4e, BP:%6.4e [grad] MG:%6.4e, BG:%6.4e", 
                               multimodal_norm_param,bias_norm_param, multimodal_norm_grad,bias_norm_grad))
      end
   end

   -- evaluate in validation set [every eval iteration or on last iteration]
   if i % opt.eval_val_every == 0 or i % iterations_per_epoch == 0 or i == iterations  then
      -- align train / val data
      vqa_data.test_data:inorder()

      local test_iter = vqa_data.test_data.iter_per_epoch
      local results = {}
      print(string.format('start evaluation on validation set'))
      for k = 1, test_iter do
         print(string.format('test -- [%d/%d]', k, test_iter))
         local imgs, x, x_len, qids = vqa_data.test_data:next_batch_image_test(vqa_data.vocab_map,
                                                       opt.cocoimg_dir, 224, 224)
         print (#imgs)
         local pred = predict_result(imgs, x, x_len)

         -- compute accuracy
         local max_score, ans = torch.max(pred, 2)
         ans = torch.reshape(ans, ans:nElement())
         for bidx, qid in pairs(qids) do
            local result = {}
            result.answer = vqa_data.answer_dict[ans[bidx]]
            result.question_id = qid
            table.insert(results, result)
         end

         if k % 10 == 0 then collectgarbage() end
      end
      if train_ex_num ~= 0 then
         train_acc = train_acc / train_ex_num
      end
      local log_apply_finetuning = 0
      if apply_finetuning then
         log_apply_finetuning = 1
      end

      testLogger:add{['iter'] = i,
                     ['epoch'] = epoch,                     
                     ['train accuracy'] = train_acc * 100,
                     ['learning_rate'] = optim_state.learningRate,
                     ['apply finetuning'] = log_apply_finetuning}

      print(string.format('iter: %d, epoch: %f,train acc: %f',
                           i, epoch,train_acc*100))

      local saveresultfile = string.format('%s/%s/vqa_%s_mscoco_%s_%s-%.2f_results.json',
                                     opt.save_dir, opt.results_dir, opt.vqa_task_type, opt.testset, opt.alg_name, epoch)
      local results_json = cjson.encode(results)
      local wf = io.open(saveresultfile, 'w')
      wf:write(results_json)
      wf:close()
      local savefile = string.format('%s/%s/snapshot_epoch%.2f.t7',
                                     opt.save_dir, opt.snapshot_dir, epoch)
      print('saving checkpoint to ' .. savefile)
      local checkpoint = {}
      checkpoint.hash_config = hash_config
      checkpoint.protos = protos
      checkpoint.i = i
      checkpoint.epoch = epoch
      checkpoint.vocab_map = vqa_data.vocab_map
      checkpoint.vocab_dict = vqa_data.vocab_dict
      checkpoint.vocab_size = vqa_data.vocab_size
      checkpoint.answer_map = vqa_data.answer_map
      checkpoint.answer_dict = vqa_data.answer_dict
      checkpoint.answer_size = vqa_data.answer_size
      checkpoint.conv_feat_dim = opt.conv_feat_dim
      checkpoint.word_dim = opt.word_dim
      checkpoint.optimizer = opt.optimizer
      checkpoint.optim_state = optim_state
      torch.save(savefile, checkpoint)

      train_acc = 0
      train_ex_num = 0
   end
   if epoch == opt.start_finetune_after then
      apply_finetuning = true
   end
   if epoch == opt.stop_finetune_after then
      apply_finetuning = false
   end
   -- exponential learning rate decay
   if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
      if epoch % opt.learning_rate_decay_after == 0 then
         local learning_rate_decay = opt.learning_rate_decay
         optim_state.learningRate = optim_state.learningRate * learning_rate_decay -- decay it
         print('decayed learning rate by a factor ' .. learning_rate_decay .. ' to ' .. optim_state.learningRate)
      end
   end  

   if i % opt.print_every == 0 then
      trainLogger:add{['iter'] = i, ['epoch'] = epoch, ['train_loss'] = train_loss, 
                      ['grad/param norm'] = grad_params:norm() / params:norm(),
                      ['time'] = time,
                      ['learning_rate'] = optim_state.learningRate}
   end

   if i % 10 == 0 then collectgarbage() end

   -- handle early stopping if things are going really bad
   if loss[1] ~= loss[1] then
      print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
      break -- halt
   end
   if loss0 == nil then loss0 = loss[1] end
   if loss[1] > loss0 * opt.loss_explod_threshold then
      print('loss is exploding, aborting.')
      break -- halt
   end
end











