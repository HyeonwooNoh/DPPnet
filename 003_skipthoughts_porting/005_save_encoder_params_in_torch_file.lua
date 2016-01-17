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
print('reading uni-GRU params ..')
local uparams = {}
uparams.U = npy4th.loadnpy(porting_data_path .. 'uparams_encoder_U.npy')
uparams.Ux = npy4th.loadnpy(porting_data_path .. 'uparams_encoder_Ux.npy')
uparams.W = npy4th.loadnpy(porting_data_path .. 'uparams_encoder_W.npy')
uparams.b = npy4th.loadnpy(porting_data_path .. 'uparams_encoder_b.npy')
uparams.Wx = npy4th.loadnpy(porting_data_path .. 'uparams_encoder_Wx.npy')
uparams.bx = npy4th.loadnpy(porting_data_path .. 'uparams_encoder_bx.npy')
print('done')
print('')

print('reading bi-GRU params ..')
local bparams = {}
bparams.U = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_U.npy')
bparams.Ux = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_Ux.npy')
bparams.W = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_W.npy')
bparams.b = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_b.npy')
bparams.Wx = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_Wx.npy')
bparams.bx = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_bx.npy')
print('done')
print('')

print('reading bi-GRU reverse params ..')
local bparams_r = {}
bparams_r.U = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_r_U.npy')
bparams_r.Ux = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_r_Ux.npy')
bparams_r.W = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_r_W.npy')
bparams_r.b = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_r_b.npy')
bparams_r.Wx = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_r_Wx.npy')
bparams_r.bx = npy4th.loadnpy(porting_data_path .. 'bparams_encoder_r_bx.npy')
print('done')
print('')

-- save uparams
print('saving uni-GRU params .. ')
torch.save(string.format('%s%s', opt.save_dir, 'uni_gru_params.t7'), uparams)
print('done')
print('')

print('saving bi-GRU params .. ')
torch.save(string.format('%s%s', opt.save_dir, 'bi_gru_params.t7'), bparams)
print('done')
print('')

print('saving bi-GRU reverse params .. ')
torch.save(string.format('%s%s', opt.save_dir, 'bi_gru_r_params.t7'), bparams_r)
print('done')
print('')









