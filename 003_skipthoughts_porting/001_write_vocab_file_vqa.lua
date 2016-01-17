local vqa_loader = require 'utils.vqa_loader'
local file_utils = require 'utils.file_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('create daquar vocab.txt file')
cmd:text()
cmd:option('-vqa_data_dir', './data/VQA_torch', 'vqa data directory')
cmd:option('-vqa_task_type', 'OpenEnded', 'task type: [OpenEnded|MultipleChoice]')
cmd:option('-vqa_ans_type', 'major', 'answer type: [major|every]')
cmd:option('-add_ans2vocab', 'false', 'option: [false|true]')
cmd:option('-seq_len', 25, 'sequence length for loading vqa data')
cmd:option('-datatype', 'test-dev2015', 'option: [trainval|test2015|test-dev2015]')
cmd:option('-batch_size', 5, 'batch size for loading vqa data')
cmd:option('-save_path', './data/skipthoughts_porting/vqa_vocab.txt', 'saving path for vqa vocab file')

-- parse input params
opt = cmd:parse(arg or {})

print('load vqa data')
-- load vqa dataset
local vqa_data = vqa_loader.load_data(opt.vqa_data_dir, opt.vqa_task_type, opt.vqa_ans_type,
                                      opt.seq_len, opt.batch_size, true, opt.datatype, 
                                      opt.batch_size, add_ans2vocab)

local vocab_dict = vqa_data.vocab_dict
print('convert vocab_dict[1] (<empty>) to UNK')
print('convert vocab_dict[2] (<unknown>) to UNK')
vocab_dict[1] = 'UNK'
vocab_dict[2] = 'UNK'

print(string.format('write vocab to file: %s',opt.save_path)) 
file_utils.write_text(opt.save_path, vocab_dict)
print('done')





