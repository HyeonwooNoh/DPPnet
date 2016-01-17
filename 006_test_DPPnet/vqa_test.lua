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
cmd:text('Evaluate DPPnet on VQA test-dev dataset')
cmd:text()
cmd:text('Options')
-- general
cmd:option('-init_from', './model/DPPnet/DPPnet_vqa.t7', 
           'initialize network parameters from checkpoint at this path')
-- data
cmd:option('-vqa_data_dir', './data/VQA_torch', 'data directory. Should contain vqa data files')
cmd:option('-cache_dir', './cache', 'directory containing caches')
cmd:option('-cocoimg_dir', './data/MSCOCO/images', 'vqa feature directory')
cmd:option('-vqa_task_type', 'OpenEnded', 'task type: [OpenEnded|MultipleChoice]')
cmd:option('-vqa_ans_type', 'major', 'answer type: [major|every]')
cmd:option('-testset', 'test-dev2015', 'option: [test-dev2015|test2015]')
cmd:option('-alg_name', 'DPPnet_vqa', 'algorithm name for vqa eval result file')
-- model params
cmd:option('-conv_feat_dim', 4096, 'dimension of extracted convnet feature')
cmd:option('-hash_size_w', 40000, 'hash size')
-- rnn params
cmd:option('-path_uni_gru_params', './data/skipthoughts_params/uni_gru_params.t7',
           'path for uni-gru parameters for skipthoughts vectors')
cmd:option('-path_uni_gru_word2vec', './data/skipthoughts_params/vqa_trainval_uni_gru_word2vec.t7',
           'path for uni-gru word2vec parameters')
-- optimization
cmd:option('-seq_len', 54, 'number of timesteps to unroll for')
cmd:option('-batch_size', 32, 'number of examples for each batch')
-- save
cmd:option('-save_dir', 'save_result_vqa_test', 'subdirectory to save results [log, snapshot]')
cmd:option('-log_dir', 'training_log', 'subdirectory to log experiments in')
cmd:option('-results_dir', 'results', 'subdirectory for saving test results')
-- bookkeeping
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-seed', 123, 'torch manual random number generator seed')
-- GPU/CPU
cmd:option('-gpuid', 0, 'which GPU to use. -1 = use CPU')

-- parse input params
opt = cmd:parse(arg or {})

torch.manualSeed(opt.seed)

-- create directory and log file
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.log_dir))
os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.results_dir))
cmd:log(string.format('%s/%s/log_cmdline', opt.save_dir, opt.log_dir), opt)
print('create directory and log file')

print('test batch size selection')
if opt.testset == 'test2015' then
   opt.test_batch_size = 38
   print('test batch size of test2015 is: 38')
elseif opt.testset == 'test-dev2015' then
   opt.test_batch_size = 32
   print('test batch size of test-dev2015 is: 32')
else
   assert(false, 'undefined testset')
end

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

print(string.format('number of data train: %d', vqa_data.train_data.ex_num_train))
print(string.format('number of data test: %d', vqa_data.test_data.ex_num_train))

-- load uni gru param and word vectors
print('load uni gru param and word vectors')
local uparams = torch.load(opt.path_uni_gru_params)
local utables = torch.load(opt.path_uni_gru_word2vec)
opt.rnn_size = uparams.Ux:size(1)
opt.word_dim = utables:size(2)

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

   protos.softmax = nn.SoftMax()
else
   assert(false, 'pre-trained model is required to run this script')
end

-- transfer to gpu
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
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
local test_acand_mask = torch.zeros(opt.test_batch_size, vqa_data.answer_size)
if opt.gpuid >= 0 then 
   test_acand_mask = test_acand_mask:float():cuda()
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
      protos.rnn:evaluate()
      protos.word_embed:evaluate()
      local we = protos.word_embed:forward(x[{t,{}}])
      local lst = protos.rnn:forward({we, rnn_state[t-1]})
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
   -- softmax
   local softmax = protos.softmax:forward(multimodal_out)

   return softmax
end

-- evaluate in validation set 
print('evaluate in validation set')
vqa_data.test_data:inorder()

local test_iter = vqa_data.test_data.iter_per_epoch
local results = {}
print(string.format('start evaluation on validation set'))
for k = 1, test_iter do
   print(string.format('test -- [%d/%d]', k, test_iter))
   local imgs, x, x_len, qids, ans_cands = vqa_data.test_data:next_batch_image_test(vqa_data.vocab_map,
                                                   opt.cocoimg_dir, 224, 224, vqa_data.answer_map)
   local pred = predict_result(imgs, x, x_len)
   if opt.vqa_task_type == 'MultipleChoice' then
      test_acand_mask:zero()
      for ak, av in pairs(ans_cands) do
         for aak, aav in pairs(av) do
            test_acand_mask[{ak, aav}] = 1
         end
      end
      pred = torch.cmul(pred, test_acand_mask)
   end
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
local saveresultfile = string.format('%s/%s/vqa_%s_mscoco_%s_%s_results.json',
                              opt.save_dir, opt.results_dir, opt.vqa_task_type, opt.testset, opt.alg_name)
local results_json = cjson.encode(results)
local wf = io.open(saveresultfile, 'w')
wf:write(results_json)
wf:close()

print('')
print('done')
