local npy4th = require 'npy4th'

cmd = torch.CmdLine()
cmd:text()
cmd:text('save params for uni-GRU, bi-GRU, word embedding in torch file')
cmd:text()
cmd:option('-porting_data_dir', './data/skipthoughts_porting/', 
           'data dir containing numpy files for skipthought params')
cmd:option('-save_dir', './data/skipthoughts_params/', 
           'directory to save torch files')

-- parse input params
opt = cmd:parse(arg or {})

print('')
print('read numpy data')
print('')
local porting_data_path = opt.porting_data_dir
print('reading uni-word embedding table ..')
local vqa_utable = npy4th.loadnpy(porting_data_path .. 'vqa_utable.npy')
print('done')
print('')

print('reading bi-word embedding table ..')
local vqa_btable = npy4th.loadnpy(porting_data_path .. 'vqa_btable.npy')
print('done')
print('')

-- create saving directory
print(string.format('create save_dir: %s', opt.save_dir))
os.execute(string.format('mkdir -p %s', opt.save_dir))

print('saving uni-GRU word embedding tables for vqa')
torch.save(string.format('%s%s', opt.save_dir, 'vqa_uni_gru_word2vec.t7'), vqa_utable)
print('done')
print('')

print('saving bi-GRU word embedding tables for vqa')
torch.save(string.format('%s%s', opt.save_dir, 'vqa_bi_gru_word2vec.t7'), vqa_btable)
print('done')
print('')


























