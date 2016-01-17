require 'image'
local stringx = require 'pl.stringx'
local file_utils = require 'utils.file_utils'
local qa_utils = require 'utils.qa_utils'
local threads = require 'threads'
local cjson = require 'cjson'

local vqa_loader = {}
vqa_loader.__index = vqa_loader

local dataclass = {}
dataclass.__index = dataclass

-- compute mean image
local mean_bgr = torch.Tensor({103.939, 116.779, 123.68})
local mean_bgr = mean_bgr:repeatTensor(224,224,1)
local mean_bgr = mean_bgr:permute(3,2,1)

function dataclass.load_data(dataSubTypes, info, vqa_dir, taskType, ansType,
                             seq_len, batch_size, opt_prefetch, add_to_vocab, add_ans2vocab, is_cached)

   local data = {}
   setmetatable(data, dataclass)
 
   -- dataSubTypes == ['train2014', 'val2014']
   data.annotations = {}
   for k, v in pairs (dataSubTypes) do
      local dataSubType = v 
      local ann_file_dir = paths.concat(string.format('%s/Annotations', vqa_dir),
                                        'mscoco_%s_annotations')
      local ann_file_path = string.format('%s/annotations.t7',
                                           string.format(ann_file_dir, dataSubType))
      data.answer_file_dir = string.format('%s/answers', ann_file_dir)
      data.question_file_dir = paths.concat(string.format('%s/Questions',vqa_dir), 
                                            taskType .. '_mscoco_%s_questions')
      -- read annotation file
      local annotation = torch.load(ann_file_path)
      local num_anno = #annotation
      for ak, av in pairs(annotation) do
         print(string.format('read annotation file [%s][%d/%d]', dataSubType, ak, num_anno))
         local ann = av
         ann.data_subtype = dataSubType
         table.insert(data.annotations,ann)
      end
   end

   -- set this data option
   data.data_subtype = table.concat(dataSubTypes)
   data.task_type = taskType
   data.ans_type = ansType

   -- set initial values
   data.ex_num_train = #data.annotations
   data.question_len = torch.zeros(data.ex_num_train)
   data.seq_len = seq_len
   data.iter_index = 0
   data.batch_index = 0
   data.batch_size = batch_size
   data.batch_order = torch.range(1, data.ex_num_train) -- in order
   data.iter_per_epoch = torch.floor(data.ex_num_train / data.batch_size)
   data.opt_prefetch = opt_prefetch
   data.opt_batch_order = 1
   data.is_trainval = false
   if data.opt_prefetch then
      -- thread for prefetching
      data.pool = threads.Threads(1)
      data.prefetch_init = false
   end

   if not is_cached then   
      -- special characters (for questions)
      if info.vocab_size == 0 then
         -- <empty>
         info.vocab_size = info.vocab_size + 1
         info.vocab_map['<empty>'] = info.vocab_size
         info.vocab_dict[info.vocab_size] = '<empty>'
         -- <unknown>
         info.vocab_size = info.vocab_size + 1
         info.vocab_map['<unknown>'] = info.vocab_size
         info.vocab_dict[info.vocab_size] = '<unknown>'
      end
      if info.answer_size == 0 then
         -- special characters (for answers)
         info.answer_size = info.answer_size + 1
         info.answer_map['<empty>'] = info.answer_size
         info.answer_dict[info.answer_size] = '<empty>'
      end
      local num_anno = #data.annotations
      for k, v in pairs(data.annotations) do
         print(string.format('[%s] reading questions: [%d/%d]', data.data_subtype, k, num_anno))
         local q_file_path = string.format('%s/%s_%d.t7', 
                                    string.format(data.question_file_dir, v.data_subtype),
                                    v.data_subtype, v.question_id)
         local question = torch.load(q_file_path)
         local tokenized = question.tokenized
         if #tokenized > info.max_sentence_len then
            info.max_sentence_len = #tokenized
         end
         data.question_len[v.ann_id] = #tokenized 
         if add_to_vocab then
            for tk, tv in pairs(tokenized) do
               local word = string.lower(tv)
               if info.vocab_map[word] == nil then
                  info.vocab_size = info.vocab_size + 1
                  info.vocab_map[word] = info.vocab_size
                  info.vocab_dict[info.vocab_size] = word
               end
            end
         end
      end
      if add_to_vocab then
         if add_ans2vocab then
            info.max_ans_len = 0
         end
         for k, v in pairs(data.annotations) do
            print(string.format('[%s] reading answers: [%d/%d]', data.data_subtype, k, num_anno))
            if ansType == 'major' then
               local ans = string.lower(v.multiple_choice_answer)
               if add_to_vocab and info.answer_map[ans] == nil then
                  info.answer_size = info.answer_size + 1
                  info.answer_map[ans] = info.answer_size
                  info.answer_dict[info.answer_size] = ans
               end
               if add_ans2vocab then
                  local ans_tok = stringx.split(ans)
                  if #ans_tok > info.max_ans_len then
                     info.max_ans_len = #ans_tok
                  end
                  for atk, atv in pairs(ans_tok) do
                     if info.vocab_map[atv] == nil then
                        info.vocab_size = info.vocab_size + 1
                        info.vocab_map[atv] = info.vocab_size
                        info.vocab_dict[info.vocab_size] = atv
                     end
                  end
               end
            elseif ansType == 'every' then
               local answer_file_path = string.format('%s/%s_%d.t7', 
                                           string.format(data.answer_file_dir,v.data_subtype),
                                           v.data_subtype, v.ann_id)
               local answers = torch.load(answer_file_path)
               for ak, av in pairs(answers) do
                  local ans = string.lower(av.answer)
                  if info.answer_map[ans] == nil then
                     info.answer_size = info.answer_size + 1
                     info.answer_map[ans] = info.answer_size
                     info.answer_dict[info.answer_size] = ans
                  end
                  if add_ans2vocab then
                     local ans_tok = stringx.split(ans)
                     if #ans_tok > info.max_ans_len then
                        info.max_ans_len = #ans_tok
                     end
                     for atk, atv in pairs(ans_tok) do
                        if info.vocab_map[atv] == nil then
                           info.vocab_size = info.vocab_size + 1
                           info.vocab_map[atv] = info.vocab_size
                           info.vocab_dict[info.vocab_size] = atv
                        end
                     end
                  end
               end
            else
               assert(false, 'undefined ansType')
            end
         end
      end
   end

   return data
end
function dataclass.load_testdata(dataSubType, 
                              info, vqa_dir, taskType,
                              seq_len, batch_size, opt_prefetch, is_cached)

   local data = {}
   setmetatable(data, dataclass)

   data.question_file_dir = paths.concat(string.format('%s/Questions',vqa_dir), 
                                         string.format('%s_mscoco_%s_questions',
                                                taskType,dataSubType))
   local question_list_path = string.format('%s/question_list.t7', 
                                  string.format(data.question_file_dir, dataSubType))
   data.question_list = torch.load(question_list_path)
   for k, v in pairs(data.question_list) do
      v.ann_id = k
      v.data_subtype = 'test2015'
   end

   -- set this data option
   data.data_subtype = dataSubType
   data.task_type = taskType

   -- set initial values
   data.ex_num_train = #data.question_list
   data.question_len = torch.zeros(data.ex_num_train)
   data.seq_len = seq_len
   data.iter_index = 0
   data.batch_index = 0
   data.batch_size = batch_size
   data.batch_order = torch.range(1, data.ex_num_train) -- in order
   data.iter_per_epoch = torch.floor(data.ex_num_train / data.batch_size)
   data.opt_prefetch = opt_prefetch
   data.opt_batch_order = 1
   if data.opt_prefetch then
      -- thread for prefetching
      data.pool = threads.Threads(1)
      data.prefetch_init = false
   end

   if not is_cached then   
      data.max_sentence_len = 0
      local num_question = #data.question_list
      for k, v in pairs(data.question_list) do
         print(string.format('read question file [test][%d/%d]', k, num_question))
         local q_file_path = string.format('%s/%s_%d.t7', 
                                 string.format(data.question_file_dir,v.data_subtype),
                                 v.data_subtype, v.question_id)
         local question = torch.load(q_file_path)
         local tokenized = question.tokenized
         if #tokenized > data.max_sentence_len then
            data.max_sentence_len = #tokenized
         end
         data.question_len[v.ann_id] = #tokenized 
      end
   end
   return data
end
function vqa_loader:qtable_to_tokens(qtable)
   local q = torch.zeros(self.seq_len,1) + 1
   local q_len = torch.zeros(1)
   for k, v in pairs(qtable) do
      if vqa_data.vocab_map[v] ~= nil then
         q[k][1] = vqa_data.vocab_map[v]
      else
         q[k][1] = vqa_data.vocab_map['<empty>']
      end
      q_len[1] = k
   end
   return q, q_len
end
function vqa_loader:question_to_tokens(str)
   local q = torch.zeros(self.seq_len,1) + 1
   local q_len = torch.zeros(1)
   local str_tokens = stringx.split(str)
   for k, v in pairs(str_tokens) do
      if vqa_data.vocab_map[v] ~= nil then
         q[k][1] = vqa_data.vocab_map[v]
      else
         q[k][1] = vqa_data.vocab_map['<empty>']
      end
      q_len[1] = k
   end
   return q, q_len
end
function vqa_loader:tokens_to_q_table_with_len(tokens, tokens_len)
   assert(tokens:nDimension() == 1, 'dimension of tokens should be 1')
   local str = {}
   for i=1,tokens_len do
      table.insert(str, self.vocab_dict[tokens[i]])
   end
   return str
end
function vqa_loader:tokens_to_question_with_len(tokens, tokens_len)
   assert(tokens:nDimension() == 1, 'dimension of tokens should be 1')
   local str = {}
   for i=1,tokens_len do
      table.insert(str, self.vocab_dict[tokens[i]])
      table.insert(str, ' ')
   end
   return table.concat(str)
end
function vqa_loader:tokens_to_question(tokens)
   assert(tokens:nDimension() == 1, 'dimension of tokens should be 1')
   local str = {}
   for i=1,tokens:size()[1] do
      table.insert(str, self.vocab_dict[tokens[i]])
      table.insert(str, ' ')
   end
   return table.concat(str)
end
function vqa_loader:tokens_to_answer(token)
   assert(type(token) == 'number')
   return self.answer_dict[token]
end
function dataclass:next_batch_feat_everya(vocab_map, ans_map, cocofeatpath, feat_dim)
   local batch_q
   local batch_q_len
   local batch_qids
   local batch_a
   local batch_feat
   local loc_feat_list
   local loc_qid_list
   local loc_subtype_list
   local loc_ans_list

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or self.prefetch_op ~= 'batch_feat' then
         self.batch_q_len = torch.zeros(self.batch_size)
         self.batch_qwords = {}
         self.batch_awords = {}
         self.height = height
         self.width = width
         self.batch_feat = torch.zeros(self.batch_size, feat_dim)
         self.feat_dim = feat_dim
         loc_ans_list = {}
         loc_subtype_list = {}
         loc_qid_list = {}
         loc_feat_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            local ann = self.annotations[ann_idx]
            local cocofeat_name = string.format('COCO_%s_%012d.t7', 
                                         ann.data_subtype, ann.image_id)
            loc_feat_list[i] = cocofeat_name
            loc_subtype_list[i] = ann.data_subtype
            loc_qid_list[i] = ann.question_id
            loc_ans_list[i] = {}
            loc_ans_list[i][1] = ann.multiple_choice_answer
            loc_ans_list[i][2] = ann.ann_id
            self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         end
         self.batch_qids = loc_qid_list
         for i = 1, self.batch_size do
            -- feature
            local feat_path = paths.concat(cocofeatpath, loc_feat_list[i]) 
            local feature = torch.load(feat_path)
            assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
            assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
            self.batch_feat[i] = feature
            -- question
            local q_file_path = string.format('%s/%s_%d.t7', 
                                    string.format(self.question_file_dir,loc_subtype_list[i]),
                                    loc_subtype_list[i], loc_qid_list[i])
            local question = torch.load(q_file_path)
            self.batch_qwords[i] = question.tokenized
            -- answer
            self.batch_awords[i] = {}
            self.batch_awords[i][1] = loc_ans_list[i][1]
            local answer_file_path = string.format('%s/%s_%d.t7',
                                            string.format(self.answer_file_dir,loc_subtype_list[i]),
                                            loc_subtype_list[i],loc_ans_list[i][2])
            local answers = torch.load(answer_file_path)
            for k, v in pairs(answers) do
               self.batch_awords[i][1+k] = v.answer
            end
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_feat_everya'
      end

      assert(self.feat_dim == feat_dim, 'feat_dim have to be save all the time')
      batch_feat = self.batch_feat:clone()
      batch_q_len = self.batch_q_len:clone()
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_a = torch.zeros(self.batch_size, 11)
      batch_qids = {}
      for k, v in pairs(self.batch_qids) do
         batch_qids[k] = v
      end
      for i = 1, self.batch_size do
         -- question
         for k, v in pairs(self.batch_qwords[i]) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         -- answer
         local ans_words = self.batch_awords[i]
         for k, v in pairs(ans_words) do
            local ans_word = string.lower(v)
            if ans_map[ans_word] == nil then
               batch_a[i][k] = ans_map['<empty>']
            else
               batch_a[i][k] = ans_map[ans_word]
            end
         end
      end
  
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_feat_list = {}
      loc_qid_list = {}
      loc_subtype_list = {}
      loc_ans_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.annotations[ann_idx]
         local cocofeat_name = string.format('COCO_%s_%012d.t7',
                                     ann.data_subtype, ann.image_id)
         loc_feat_list[i] = cocofeat_name
         loc_qid_list[i] = ann.question_id
         loc_subtype_list[i] = ann.data_subtype
         loc_ans_list[i] = {}
         loc_ans_list[i][1] = ann.multiple_choice_answer
         loc_ans_list[i][2] = ann.ann_id
         self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
      end
      self.batch_qids = loc_qid_list
      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      local loc_question_file_dir = self.question_file_dir
      local loc_data_subtype = self.data_subtype
      local loc_answer_file_dir = self.answer_file_dir
      local loc_ans_type = self.ans_type
      self.pool:addjob(
         function ()
            local pre_feature = torch.zeros(loc_batch_size, feat_dim)
            local pre_batch_qwords = {}
            local pre_batch_awords = {}
            for i = 1, loc_batch_size do
               local feat_path = paths.concat(cocofeatpath, loc_feat_list[i])
               local feature = torch.load(feat_path)
               assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
               assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
               pre_feature[i] = feature
               -- question
               local q_file_path = string.format('%s/%s_%d.t7', 
                                       string.format(loc_question_file_dir, loc_subtype_list[i]),
                                       loc_subtype_list[i], loc_qid_list[i])
               local question = torch.load(q_file_path)
               pre_batch_qwords[i] = question.tokenized
               -- answer
               pre_batch_awords[i] = {}
               pre_batch_awords[i][1] = loc_ans_list[i][1]
               local answer_file_path = string.format('%s/%s_%d.t7',
                                            string.format(loc_answer_file_dir,loc_subtype_list[i]),
                                            loc_subtype_list[i],loc_ans_list[i][2])
               local answers = torch.load(answer_file_path)
               for k, v in pairs(answers) do
                  pre_batch_awords[i][1+k] = v.answer
               end
            end
            return pre_feature, pre_batch_qwords, pre_batch_awords
         end,
         function (pre_feature, pre_batch_qwords, pre_batch_awords)
            self.batch_feat = pre_feature
            self.batch_qwords = pre_batch_qwords
            self.batch_awords = pre_batch_awords
         end
      )
   else
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_q_len = torch.zeros(self.batch_size)
      batch_qids = {}
      batch_a = torch.zeros(self.batch_size,11)
      batch_feat = torch.zeros(self.batch_size, feat_dim)
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.annotations[ann_idx]
         local cocofeat_name = string.format('COCO_%s_%012d.t7',
                                     ann.data_subtype, ann.image_id)

         local feat_path = paths.concat(cocofeatpath, cocofeat_name)
         local feature = torch.load(feat_path)
         assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
         assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
         batch_feat[i] = feature
         batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         batch_qids[i] = ann.question_id
         local q_file_path = string.format('%s/%s_%d.t7', 
                                 string.format(self.question_file_dir,ann.data_subtype),
                                 ann.data_subtype, ann.question_id)
         local question = torch.load(q_file_path)
         for k, v in pairs(question.tokenized) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         local major_ans_word = string.lower(self.ann.multiple_choice_answer)
         if ans_map[major_ans_word] == nil then
            batch_a[i][1] = ans_map['<empty>']
         else
            batch_a[i][1] = ans_map[major_ans_word]
         end
         local answer_file_path = string.format('%s/%s_%d.t7',
                                         string.format(self.answer_file_dir,ann.data_subtype),
                                         ann.data_subtype, ann.ann_id)
         local answers = torch.load(answer_file_path)
         for k, v in pairs(answers) do
            local ans_word = string.lower(v.answer)
            if ans_map[ans_word] == nil then
               batch_a[i] = ans_map['<empty>']
            else
               batch_a[i] = ans_map[ans_word]
            end
         end
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_feat:clone(), batch_q:transpose(1,2):clone(), batch_q_len:clone(), 
          batch_a:clone(), batch_qids
end
function dataclass:next_batch_feat(vocab_map, ans_map, cocofeatpath, feat_dim)
   local batch_q
   local batch_q_len
   local batch_a
   local batch_feat
   local loc_feat_list
   local loc_qid_list
   local loc_subtype_list
   local loc_ans_list

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or self.prefetch_op ~= 'batch_feat' then
         self.batch_q_len = torch.zeros(self.batch_size)
         self.batch_qwords = {}
         self.batch_awords = {}
         self.height = height
         self.width = width
         self.batch_feat = torch.zeros(self.batch_size, feat_dim)
         self.feat_dim = feat_dim
         loc_ans_list = {}
         loc_subtype_list = {}
         loc_qid_list = {}
         loc_feat_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            local ann = self.annotations[ann_idx]
            local cocofeat_name = string.format('COCO_%s_%012d.t7', 
                                        ann.data_subtype, ann.image_id)
            loc_feat_list[i] = cocofeat_name
            loc_subtype_list[i] = ann.data_subtype
            loc_qid_list[i] = ann.question_id
            if self.ans_type == 'major' then
               loc_ans_list[i] = ann.multiple_choice_answer
            elseif self.ans_type == 'every' then
               loc_ans_list[i] = ann.ann_id
            else
               assert(false, 'undefined answer type')
            end
            self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         end
         for i = 1, self.batch_size do
            -- feature
            local feat_path = paths.concat(cocofeatpath, loc_feat_list[i]) 
            local feature = torch.load(feat_path)
            assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
            assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
            self.batch_feat[i] = feature
            -- question
            local q_file_path = string.format('%s/%s_%d.t7', 
                                    string.format(self.question_file_dir,loc_subtype_list[i]),
                                    loc_subtype_list[i], loc_qid_list[i])
            local question = torch.load(q_file_path)
            self.batch_qwords[i] = question.tokenized
            -- answer
            if self.ans_type == 'major' then
               self.batch_awords[i] = loc_ans_list[i]
            elseif self.ans_type == 'every' then
               local answer_file_path = string.format('%s/%s_%d.t7',
                                             string.format(self.answer_file_dir,loc_subtype_list[i]),
                                             loc_subtype_list[i],loc_ans_list[i])
               local answers = torch.load(answer_file_path)
               local rand_n = (torch.random() % (#answers)) + 1
               self.batch_awords[i] = answers[rand_n].answer
            else
               assert(false, 'undefined answer type')
            end
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_feat'
      end

      assert(self.feat_dim == feat_dim, 'feat_dim have to be save all the time')
      batch_feat = self.batch_feat:clone()
      batch_q_len = self.batch_q_len:clone()
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_a = torch.zeros(self.batch_size)
      for i = 1, self.batch_size do
         -- question
         for k, v in pairs(self.batch_qwords[i]) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         -- answer
         local ans_word = string.lower(self.batch_awords[i])
         if ans_map[ans_word] == nil then
            batch_a[i] = ans_map['<empty>']
         else
            batch_a[i] = ans_map[ans_word]
         end
      end
  
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_feat_list = {}
      loc_subtype_list = {}
      loc_qid_list = {}
      loc_ans_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.annotations[ann_idx]
         local cocofeat_name = string.format('COCO_%s_%012d.t7',
                                     ann.data_subtype, ann.image_id)
         loc_feat_list[i] = cocofeat_name
         loc_subtype_list[i] = ann.data_subtype
         loc_qid_list[i] = ann.question_id
         if self.ans_type == 'major' then
            loc_ans_list[i] = ann.multiple_choice_answer
         elseif self.ans_type == 'every' then
            loc_ans_list[i] = ann.ann_id
         else
            assert(false, 'undefined answer type')
         end
         self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
      end
      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      local loc_question_file_dir = self.question_file_dir
      local loc_answer_file_dir = self.answer_file_dir
      local loc_ans_type = self.ans_type
      self.pool:addjob(
         function ()
            local pre_feature = torch.zeros(loc_batch_size, feat_dim)
            local pre_batch_qwords = {}
            local pre_batch_awords = {}
            for i = 1, loc_batch_size do
               local feat_path = paths.concat(cocofeatpath, loc_feat_list[i])
               local feature = torch.load(feat_path)
               assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
               assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
               pre_feature[i] = feature
               -- question
               local q_file_path = string.format('%s/%s_%d.t7',
                                       string.format(loc_question_file_dir,loc_subtype_list[i]),
                                       loc_subtype_list[i], loc_qid_list[i])
               local question = torch.load(q_file_path)
               pre_batch_qwords[i] = question.tokenized
               -- answer
               if loc_ans_type == 'major' then
                  pre_batch_awords[i] = loc_ans_list[i]
               elseif loc_ans_type == 'every' then
                  local answer_file_path = string.format('%s/%s_%d.t7',
                                               string.format(loc_answer_file_dir,loc_subtype_list[i]),                                               loc_subtype_list[i],loc_ans_list[i])
                  local answers = torch.load(answer_file_path)
                  local rand_n = (torch.random() % (#answers)) + 1
                  pre_batch_awords[i] = answers[rand_n].answer
               else
                  assert(false, 'undefined answer type')
               end
            end
            return pre_feature, pre_batch_qwords, pre_batch_awords
         end,
         function (pre_feature, pre_batch_qwords, pre_batch_awords)
            self.batch_feat = pre_feature
            self.batch_qwords = pre_batch_qwords
            self.batch_awords = pre_batch_awords
         end
      )
   else
      local batch_q = torch.zeros(self.batch_size, self.seq_len)
      local batch_q_len = torch.zeros(self.batch_size)
      local batch_a = torch.zeros(self.batch_size)
      local batch_feat = torch.zeros(self.batch_size, feat_dim)
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.annotations[ann_idx]
         local cocofeat_name = string.format('COCO_%s_%012d.t7',
                                     ann.data_subtype, ann.image_id)

         local feat_path = paths.concat(cocofeatpath, cocofeat_name)
         local feature = torch.load(feat_path)
         assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
         assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
         batch_feat[i] = feature
         batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]

         local q_file_path = string.format('%s/%s_%d.t7', 
                                 string.format(self.question_file_dir,ann.data_subtype),
                                 ann.data_subtype, ann.question_id)
         local question = torch.load(q_file_path)
         for k, v in pairs(question.tokenized) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         if self.ans_type == 'major' then
            local ans_word = string.lower(ann.multiple_choice_answer)
            if ans_map[ans_word] == nil then
               batch_a[i] = ans_map['<empty>']
            else
               batch_a[i] = ans_map[ans_word]
            end
         elseif self.ans_type == 'every' then
            local answer_file_path = string.format('%s/%s_%d.t7',
                                            string.format(self.answer_file_dir,ann.data_subtype),
                                            ann.data_subtype,ann.ann_id)
            local answers = torch.load(answer_file_path)
            local rand_n = (torch.random() % (#answers)) + 1
            local ans_word = string.lower(answers[rand_n].answer)
            if ans_map[ans_word] == nil then
               batch_a[i] = ans_map['<empty>']
            else
               batch_a[i] = ans_map[ans_word]
            end
         else
            assert(false, 'undefined answer type')
         end
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_feat:clone(), batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone()
end
function dataclass:next_batch_feat_test(vocab_map, cocofeatpath, feat_dim, ans_map)
   local batch_q
   local batch_q_len
   local batch_qids
   local batch_feat
   local batch_a_cands
   local loc_feat_list
   local loc_qid_list
   local loc_subtype_list
   local loc_task_type = self.task_type

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or self.prefetch_op ~= 'batch_feat_test' then
         self.batch_q_len = torch.zeros(self.batch_size)
         self.batch_qids = {}
         self.batch_qwords = {}
         self.batch_a_candwords = {}
         self.batch_feat = torch.zeros(self.batch_size, feat_dim)
         self.feat_dim = feat_dim
         loc_subtype_list = {}
         loc_qid_list = {}
         loc_feat_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            local ann = self.question_list[ann_idx]
            local cocofeat_name = string.format('COCO_%s_%012d.t7', 
                                        ann.data_subtype, ann.image_id)
            loc_feat_list[i] = cocofeat_name
            loc_subtype_list[i] = ann.data_subtype
            loc_qid_list[i] = ann.question_id
            self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         end
         self.batch_qids = loc_qid_list
         for i = 1, self.batch_size do
            -- feature
            local feat_path = paths.concat(cocofeatpath, loc_feat_list[i]) 
            local feature = torch.load(feat_path)
            assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
            assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
            self.batch_feat[i] = feature
            -- question
            local q_file_path = string.format('%s/%s_%d.t7', 
                                    self.question_file_dir,
                                    loc_subtype_list[i], loc_qid_list[i])
            local question = torch.load(q_file_path)
            self.batch_qwords[i] = question.tokenized
            if loc_task_type == 'MultipleChoice' then
               self.batch_a_candwords[i] = question.multiple_choices
            end
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_feat_test'
      end

      assert(self.feat_dim == feat_dim, 'feat_dim have to be save all the time')
      batch_feat = self.batch_feat:clone()
      batch_q_len = self.batch_q_len:clone()
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_a_cands = {}
      batch_qids = {}
      for k, v in pairs(self.batch_qids) do
         batch_qids[k] = v
      end
      for i = 1, self.batch_size do
         -- question
         for k, v in pairs(self.batch_qwords[i]) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         if loc_task_type == 'MultipleChoice' then
            -- answer candidates
            batch_a_cands[i] = {}
            for k, v in pairs(self.batch_a_candwords[i]) do
               local ans_word = string.lower(v)
               if ans_map[ans_word] == nil then
                  batch_a_cands[i][k] = ans_map['<empty>']
              else
                  batch_a_cands[i][k] = ans_map[ans_word]
               end
            end
         end
      end
  
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_feat_list = {}
      loc_subtype_list = {}
      loc_qid_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.question_list[ann_idx]
         local cocofeat_name = string.format('COCO_%s_%012d.t7',
                                     ann.data_subtype, ann.image_id)
         loc_feat_list[i] = cocofeat_name
         loc_subtype_list[i] = ann.data_subtype
         loc_qid_list[i] = ann.question_id
         self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
      end
      self.batch_qids = loc_qid_list
      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      local loc_question_file_dir = self.question_file_dir
      self.pool:addjob(
         function ()
            local pre_feature = torch.zeros(loc_batch_size, feat_dim)
            local pre_batch_qwords = {}
            local pre_batch_a_candwords = {} 
            for i = 1, loc_batch_size do
               local feat_path = paths.concat(cocofeatpath, loc_feat_list[i])
               local feature = torch.load(feat_path)
               assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
               assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
               pre_feature[i] = feature
               -- question
               local q_file_path = string.format('%s/%s_%d.t7',
                                       loc_question_file_dir,
                                       loc_subtype_list[i], loc_qid_list[i])
               local question = torch.load(q_file_path)
               pre_batch_qwords[i] = question.tokenized
               if loc_task_type == 'MultipleChoice' then
                  pre_batch_a_candwords[i] = question.multiple_choices
               end
            end
            return pre_feature, pre_batch_qwords, pre_batch_a_candwords
         end,
         function (pre_feature, pre_batch_qwords, pre_batch_a_candwords)
            self.batch_feat = pre_feature
            self.batch_qwords = pre_batch_qwords
            self.batch_a_candwords = pre_batch_a_candwords
         end
      )
   else
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_q_len = torch.zeros(self.batch_size)
      batch_feat = torch.zeros(self.batch_size, feat_dim)
      batch_qids = {}
      batch_a_cands = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.question_list[ann_idx]
         local cocofeat_name = string.format('COCO_%s_%012d.t7',
                                     ann.data_subtype, ann.image_id)
         local feat_path = paths.concat(cocofeatpath, cocofeat_name)
         local feature = torch.load(feat_path)
         assert(feature:dim() == 1, 'only 1 dimension feature could be loaded with this method')
         assert(feature:nElement() == feat_dim, string.format('dimension mismatch: dimension of saved feature is: %d', feature:nElement()))
         batch_feat[i] = feature
         batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]

         batch_qids[i] = ann.question_id
         local q_file_path = string.format('%s/%s_%d.t7', 
                                 self.question_file_dir,
                                 ann.data_subtype, ann.question_id)
         local question = torch.load(q_file_path)
         for k, v in pairs(question.tokenized) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         if loc_task_type == 'MultipleChoice' then
            -- answer candidates
            batch_a_cands[i] = {}
            for k, v in pairs(question.multiple_choices) do
               local ans_word = string.lower(v)
               if ans_map[ans_word] == nil then
                  batch_a_cands[i][k] = ans_map['<empty>']
               else
                  batch_a_cands[i][k] = ans_map[ans_word]
               end
            end
         end
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_feat:clone(), batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_qids, batch_a_cands
end


function dataclass:next_batch_image_test(vocab_map, cocoimgpath, height, width, ans_map)
   local batch_q
   local batch_q_len
   local batch_qids
   local batch_img
   local batch_a_cands
   local loc_img_list
   local loc_qid_list
   local loc_subtype_list
   local loc_task_type = self.task_type

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or self.prefetch_op ~= 'batch_image_test' then
         self.batch_q_len = torch.zeros(self.batch_size)
         self.batch_qwords = {}
         self.batch_a_candwords = {}
         self.batch_qids = {}
         self.height = height
         self.width = width
         self.batch_img = torch.zeros(self.batch_size, 3, height, width)
         loc_subtype_list = {}
         loc_qid_list = {}
         loc_img_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            local ann = self.question_list[ann_idx]
            local cocoimg_name = string.format('COCO_%s_%012d.jpg', 
                                        ann.data_subtype, ann.image_id)
            loc_img_list[i] = cocoimg_name
            loc_subtype_list[i] = ann.data_subtype
            loc_qid_list[i] = ann.question_id
            self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         end
         self.batch_qids = loc_qid_list
         for i = 1, self.batch_size do
            -- image
            local img_path = paths.concat(string.format('%s/%s',
                                                 cocoimgpath,loc_subtype_list[i]), 
                                          loc_img_list[i]) 
            local img = image.load(img_path)
            img = image.scale(img, width, height)
            if img:size()[1] == 1 then
               img = img:repeatTensor(3,1,1)
            end
            img = img:index(1, torch.LongTensor{3,2,1})
            img = img * 255 - mean_bgr
            img = img:contiguous()

            self.batch_img[i] = img
            -- question
            local q_file_path = string.format('%s/%s_%d.t7', 
                                         self.question_file_dir,
                                         loc_subtype_list[i], loc_qid_list[i])
            local question = torch.load(q_file_path)
            self.batch_qwords[i] = question.tokenized
            if loc_task_type == 'MultipleChoice' then
               self.batch_a_candwords[i] = question.multiple_choices
            end
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_image_test'
      end
      assert(self.height == height, 'height have to be same all the time')
      assert(self.width == width, 'width have to be same all the time')
      batch_img = self.batch_img:clone()
      batch_q_len = self.batch_q_len:clone()
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_a_cands = {}
      batch_qids = {}
      for k, v in pairs(self.batch_qids) do
         batch_qids[k] = v
      end
      for i = 1, self.batch_size do
         -- question
         for k, v in pairs(self.batch_qwords[i]) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         if loc_task_type == 'MultipleChoice' then
            -- answer candidates
            batch_a_cands[i] = {}
            for k, v in pairs(self.batch_a_candwords[i]) do
               local ans_word = string.lower(v)
               if ans_map[ans_word] == nil then
                  batch_a_cands[i][k] = ans_map['<empty>']
               else
                  batch_a_cands[i][k] = ans_map[ans_word]
               end
            end
         end
      end
  
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_img_list = {}
      loc_subtype_list = {}
      loc_qid_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.question_list[ann_idx]
         local cocoimg_name = string.format('COCO_%s_%012d.jpg',
                                     ann.data_subtype, ann.image_id)
         loc_img_list[i] = cocoimg_name
         loc_subtype_list[i] = ann.data_subtype
         loc_qid_list[i] = ann.question_id
         self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
      end
      self.batch_qids = loc_qid_list
      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      local loc_question_file_dir = self.question_file_dir
      self.pool:addjob(
         function ()
            local loc_image = require 'image'
            local pre_img = torch.zeros(loc_batch_size, 3, height, width)
            local pre_batch_qwords = {}
            local pre_batch_a_candwords = {}
            for i = 1, loc_batch_size do
               local img_path = paths.concat(string.format('%s/%s',
                                                    cocoimgpath,loc_subtype_list[i]),
                                             loc_img_list[i]) 
               local img = loc_image.load(img_path)
               img = loc_image.scale(img, width, height)
               if img:size()[1] == 1 then
                  img = img:repeatTensor(3,1,1)
               end
               img = img:index(1, torch.LongTensor{3,2,1})
               img = img * 255 - mean_bgr
               img = img:contiguous()
               pre_img[i] = img
               -- question
               local q_file_path = string.format('%s/%s_%d.t7', 
                                       loc_question_file_dir,
                                       loc_subtype_list[i], loc_qid_list[i])
               local question = torch.load(q_file_path)
               pre_batch_qwords[i] = question.tokenized
               if loc_task_type == 'MultipleChoice' then
                  pre_batch_a_candwords[i] = question.multiple_choices
               end
            end
            return pre_img, pre_batch_qwords, pre_batch_a_candwords
         end,
         function (pre_img, pre_batch_qwords, pre_batch_a_candwords)
            self.batch_img = pre_img
            self.batch_qwords = pre_batch_qwords
            self.batch_a_candwords = pre_batch_a_candwords
         end
      )
   else
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_q_len = torch.zeros(self.batch_size)
      batch_img = torch.zeros(self.batch_size, 3, height, width)
      batch_qids = {}
      batch_a_cands = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.question_list[ann_idx]
         local cocoimg_name = string.format('COCO_%s_%012d.jpg',
                                     ann.data_subtype, ann.image_id)
         local img_path = paths.concat(string.format('%s/%s',
                                              cocoimgpath,ann.data_subtype),
                                       loc_img_list[i])
         local img = image.load(img_path)
         img = image.scale(img, width, height)
         if img:size()[1] == 1 then
            img = img:repeatTensor(3,1,1)
         end
         img = img:index(1, torch.LongTensor{3,2,1})
         img = img * 255 - mean_bgr
         img = img:contiguous()
         batch_img[i] = img
         batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         batch_qids[i] = ann.question_id

         local q_file_path = string.format('%s/%s_%d.t7',
                                 self.question_file_dir,
                                 ann.data_subtype, ann.question_id)
         local question = torch.load(q_file_path)
         for k, v in pairs(question.tokenized) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         if loc_task_type == 'MultipleChoice' then
            -- answer candidates
            batch_a_cands[i] = {}
            for k, v in pairs(question.multiple_choices) do
               local ans_word = string.lower(v)
               if ans_map[ans_word] == nil then
                  batch_a_cands[i][k] = ans_map['<empty>']
               else
                  batch_a_cands[i][k] = ans_map[ans_word]
               end
            end
         end
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_img:clone(), batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_qids, batch_a_cands
end
function dataclass:next_batch_image_sal(vocab_map, ans_map, cocoimgpath, height, width)
   local batch_q
   local batch_q_len
   local batch_w
   local batch_h
   local batch_a
   local batch_img
   local batch_qids
   local loc_img_list
   local loc_qid_list
   local loc_subtype_list
   local loc_ans_list

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or self.prefetch_op ~= 'batch_image' then
         self.batch_q_len = torch.zeros(self.batch_size)
         self.batch_w = torch.zeros(self.batch_size)
         self.batch_h = torch.zeros(self.batch_size)
         self.batch_qwords = {}
         self.batch_qids = {}
         self.batch_awords = {}
         self.height = height
         self.width = width
         self.batch_img = torch.zeros(self.batch_size, 3, height, width)
         loc_ans_list = {}
         loc_subtype_list = {}
         loc_qid_list = {}
         loc_img_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            local ann = self.annotations[ann_idx]
            local cocoimg_name = string.format('COCO_%s_%012d.jpg', 
                                        ann.data_subtype, ann.image_id)
            loc_img_list[i] = cocoimg_name
            loc_subtype_list[i] = ann.data_subtype
            loc_qid_list[i] = ann.question_id
            if self.ans_type == 'major' then
               loc_ans_list[i] = ann.multiple_choice_answer
            elseif self.ans_type == 'every' then
               loc_ans_list[i] = ann.ann_id
            else
               assert(false, 'undefined answer type')
            end
            self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         end
         self.batch_qids = loc_qid_list
         for i = 1, self.batch_size do
            -- image
            local img_path = paths.concat(string.format('%s/%s',
                                                 cocoimgpath,loc_subtype_list[i]), 
                                          loc_img_list[i]) 
            local img = image.load(img_path)
            self.batch_h[i] = (#img)[2]
            self.batch_w[i] = (#img)[3]
            img = image.scale(img, width, height)
            if img:size()[1] == 1 then
               img = img:repeatTensor(3,1,1)
            end
            img = img:index(1, torch.LongTensor{3,2,1})
            img = img * 255 - mean_bgr
            img = img:contiguous()
            self.batch_img[i] = img
            -- question
            local q_file_path = string.format('%s/%s_%d.t7', 
                                    string.format(self.question_file_dir,loc_subtype_list[i]),
                                         loc_subtype_list[i], loc_qid_list[i])
            local question = torch.load(q_file_path)
            self.batch_qwords[i] = question.tokenized
            -- answer
            if self.ans_type == 'major' then
               self.batch_awords[i] = loc_ans_list[i]
            elseif self.ans_type == 'every' then
               local answer_file_path = string.format('%s/%s_%d.t7',
                                           string.format(self.answer_file_dir,loc_subtype_list[i]),
                                           loc_subtype_list[i],loc_ans_list[i])
               local answers = torch.load(answer_file_path)
               local rand_n = (torch.random() % (#answers)) + 1
               self.batch_awords[i] = answers[rand_n].answer
            else
               assert(false, 'undefined answer type')
            end
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_image'
      end
      assert(self.height == height, 'height have to be same all the time')
      assert(self.width == width, 'width have to be same all the time')
      batch_img = self.batch_img:clone()
      batch_q_len = self.batch_q_len:clone()
      batch_w = self.batch_w:clone()
      batch_h = self.batch_h:clone()
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_a = torch.zeros(self.batch_size)
      batch_qids = {}
      for k, v in pairs(self.batch_qids) do
         batch_qids[k] = v
      end
      for i = 1, self.batch_size do
         -- question
         for k, v in pairs(self.batch_qwords[i]) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         -- answer
         local ans_word = string.lower(self.batch_awords[i])
         if ans_map[ans_word] == nil then
            batch_a[i] = ans_map['<empty>']
         else
            batch_a[i] = ans_map[ans_word]
         end
      end
  
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_img_list = {}
      loc_subtype_list = {}
      loc_qid_list = {}
      loc_ans_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.annotations[ann_idx]
         local cocoimg_name = string.format('COCO_%s_%012d.jpg',
                                     ann.data_subtype, ann.image_id)
         loc_img_list[i] = cocoimg_name
         loc_subtype_list[i] = ann.data_subtype
         loc_qid_list[i] = ann.question_id
         if self.ans_type == 'major' then
            loc_ans_list[i] = ann.multiple_choice_answer
         elseif self.ans_type == 'every' then
            loc_ans_list[i] = ann.ann_id
         else
            assert(false, 'undefined answer type')
         end
         self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
      end
      self.batch_qids = loc_qid_list
      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      local loc_question_file_dir = self.question_file_dir
      local loc_answer_file_dir = self.answer_file_dir
      local loc_ans_type = self.ans_type
      self.pool:addjob(
         function ()
            local loc_image = require 'image'
            local pre_img = torch.zeros(loc_batch_size, 3, height, width)
            local pre_w = torch.zeros(loc_batch_size)
            local pre_h = torch.zeros(loc_batch_size)
            local pre_batch_qwords = {}
            local pre_batch_awords = {}
            for i = 1, loc_batch_size do
               local img_path = paths.concat(string.format('%s/%s',
                                                    cocoimgpath,loc_subtype_list[i]),
                                             loc_img_list[i]) 
               local img = loc_image.load(img_path)
               pre_h[i] = (#img)[2]
               pre_w[i] = (#img)[3]
               img = loc_image.scale(img, width, height)
               if img:size()[1] == 1 then
                  img = img:repeatTensor(3,1,1)
               end
               img = img:index(1, torch.LongTensor{3,2,1})
               img = img * 255 - mean_bgr
               img = img:contiguous()
               pre_img[i] = img
               -- question
               local q_file_path = string.format('%s/%s_%d.t7', 
                                       string.format(loc_question_file_dir,loc_subtype_list[i]),
                                       loc_subtype_list[i], loc_qid_list[i])
               local question = torch.load(q_file_path)
               pre_batch_qwords[i] = question.tokenized
               -- answer
               if loc_ans_type == 'major' then
                  pre_batch_awords[i] = loc_ans_list[i]
               elseif loc_ans_type == 'every' then
                  local answer_file_path = string.format('%s/%s_%d.t7',
                                              string.format(loc_answer_file_dir,loc_subtype_list[i]),
                                              loc_subtype_list[i],loc_ans_list[i])
                  local answers = torch.load(answer_file_path)
                  local rand_n = (torch.random() % (#answers)) + 1
                  pre_batch_awords[i] = answers[rand_n].answer
               else
                  assert(false, 'undefined answer type')
               end
            end
            return pre_img, pre_batch_qwords, pre_batch_awords, pre_w, pre_h
         end,
         function (pre_img, pre_batch_qwords, pre_batch_awords, pre_w, pre_h)
            self.batch_img = pre_img
            self.batch_w = pre_w
            self.batch_h = pre_h
            self.batch_qwords = pre_batch_qwords
            self.batch_awords = pre_batch_awords
         end
      )
   else
      local batch_q = torch.zeros(self.batch_size, self.seq_len)
      local batch_q_len = torch.zeros(self.batch_size)
      local batch_h = torch.zeros(self.batch_size)
      local batch_w = torch.zeros(self.batch_size)
      local batch_a = torch.zeros(self.batch_size)
      local batch_img = torch.zeros(self.batch_size, 3, height, width)
      local batch_qids = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.annotations[ann_idx]
         local cocoimg_name = string.format('COCO_%s_%012d.jpg',
                                     ann.data_subtype, ann.image_id)
         local img_path = paths.concat(string.format('%s/%s',
                                              cocoimgpath,ann.data_subtype),
                                       loc_img_list[i])
         local img = image.load(img_path)
         batch_h[i] = (#img)[2]
         batch_w[i] = (#img)[3]
         img = image.scale(img, width, height)
         if img:size()[1] == 1 then
            img = img:repeatTensor(3,1,1)
         end
         img = img:index(1, torch.LongTensor{3,2,1})
         img = img * 255 - mean_bgr
         img = img:contiguous()
         batch_img[i] = img
         batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         batch_qids[i] = ann.question_id

         local q_file_path = string.format('%s/%s_%d.t7',
                                 string.format(self.question_file_dir,ann.data_subtype),
                                 ann.data_subtype, ann.question_id)
         local question = torch.load(q_file_path)
         for k, v in pairs(question.tokenized) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         if self.ans_type == 'major' then
            local ans_word = string.lower(ann.multiple_choice_answer)
            if ans_map[ans_word] == nil then
               batch_a[i] = ans_map['<empty>']
            else
               batch_a[i] = ans_map[ans_word]
            end
         elseif self.ans_type == 'every' then
            local answer_file_path = string.format('%s/%s_%d.t7',
                                         string.format(self.answer_file_dir,ann.data_subtype),
                                         ann.data_subtype,ann.ann_id)
            local answers = torch.load(answer_file_path)
            local rand_n = (torch.random() % (#answers)) + 1
            local ans_word = string.lower(answers[rand_n].answer)
            if ans_map[ans_word] == nil then
               batch_a[i] = ans_map['<empty>']
            else
               batch_a[i] = ans_map[ans_word]
            end
         else
            assert(false, 'undefined answer type')
         end
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_img:clone(), batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone(), batch_qids, batch_w, batch_h
end
function dataclass:next_batch_image(vocab_map, ans_map, cocoimgpath, height, width,
                                    randcrop, reheight, rewidth)
   local batch_q
   local batch_q_len
   local batch_a
   local batch_img
   local loc_img_list
   local loc_qid_list
   local loc_subtype_list
   local loc_ans_list

   if randcrop == nil then
      randcrop = false
   end

   if self.opt_prefetch then
      self.pool:synchronize()
      if self.prefetch_init == false or 
         self.prefetch_op ~= 'batch_image' or
         self.randcrop ~= randcrop then

         self.batch_q_len = torch.zeros(self.batch_size)
         self.batch_qwords = {}
         self.batch_awords = {}
         self.height = height
         self.width = width
         self.batch_img = torch.zeros(self.batch_size, 3, height, width)
         self.randcrop = randcrop    

         loc_ans_list = {}
         loc_subtype_list = {}
         loc_qid_list = {}
         loc_img_list = {}
         for i = 1, self.batch_size do
            local ann_idx = self.batch_order[i + self.batch_index]
            local ann = self.annotations[ann_idx]
            local cocoimg_name = string.format('COCO_%s_%012d.jpg', 
                                        ann.data_subtype, ann.image_id)
            loc_img_list[i] = cocoimg_name
            loc_subtype_list[i] = ann.data_subtype
            loc_qid_list[i] = ann.question_id
            if self.ans_type == 'major' then
               loc_ans_list[i] = ann.multiple_choice_answer
            elseif self.ans_type == 'every' then
               loc_ans_list[i] = ann.ann_id
            else
               assert(false, 'undefined answer type')
            end
            self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
         end
         for i = 1, self.batch_size do
            -- image
            local img_path = paths.concat(string.format('%s/%s',
                                                 cocoimgpath,loc_subtype_list[i]), 
                                          loc_img_list[i]) 
            local img = image.load(img_path)
            if randcrop then
               img = image.scale(img, rewidth, reheight)
            else
               img = image.scale(img, width, height)
            end
            if img:size()[1] == 1 then
               img = img:repeatTensor(3,1,1)
            end
            if randcrop then
               local cx1 = torch.random() % (rewidth-width) + 1
               local cy1 = torch.random() % (reheight-height) + 1
               local cx2 = cx1 + width
               local cy2 = cy1 + height
               img = image.crop(img, cx1, cy1, cx2, cy2)
            end
            img = img:index(1, torch.LongTensor{3,2,1})
            img = img * 255 - mean_bgr
            img = img:contiguous()
            self.batch_img[i] = img
            -- question
            local q_file_path = string.format('%s/%s_%d.t7', 
                                    string.format(self.question_file_dir,loc_subtype_list[i]),
                                         loc_subtype_list[i], loc_qid_list[i])
            local question = torch.load(q_file_path)
            self.batch_qwords[i] = question.tokenized
            -- answer
            if self.ans_type == 'major' then
               self.batch_awords[i] = loc_ans_list[i]
            elseif self.ans_type == 'every' then
               local answer_file_path = string.format('%s/%s_%d.t7',
                                           string.format(self.answer_file_dir,loc_subtype_list[i]),
                                           loc_subtype_list[i],loc_ans_list[i])
               local answers = torch.load(answer_file_path)
               local rand_n = (torch.random() % (#answers)) + 1
               self.batch_awords[i] = answers[rand_n].answer
            else
               assert(false, 'undefined answer type')
            end
         end
         
         self.prefetch_init = true
         self.prefetch_op = 'batch_image'
      end
      assert(self.height == height, 'height have to be same all the time')
      assert(self.width == width, 'width have to be same all the time')
      batch_img = self.batch_img:clone()
      batch_q_len = self.batch_q_len:clone()
      batch_q = torch.zeros(self.batch_size, self.seq_len)
      batch_a = torch.zeros(self.batch_size)
      for i = 1, self.batch_size do
         -- question
         for k, v in pairs(self.batch_qwords[i]) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         -- answer
         local ans_word = string.lower(self.batch_awords[i])
         if ans_map[ans_word] == nil then
            batch_a[i] = ans_map['<empty>']
         else
            batch_a[i] = ans_map[ans_word]
         end
      end
  
      -- update batch counter
      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
  
      -- light weight fetching
      loc_img_list = {}
      loc_subtype_list = {}
      loc_qid_list = {}
      loc_ans_list = {}
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.annotations[ann_idx]
         local cocoimg_name = string.format('COCO_%s_%012d.jpg',
                                     ann.data_subtype, ann.image_id)
         loc_img_list[i] = cocoimg_name
         loc_subtype_list[i] = ann.data_subtype
         loc_qid_list[i] = ann.question_id
         if self.ans_type == 'major' then
            loc_ans_list[i] = ann.multiple_choice_answer
         elseif self.ans_type == 'every' then
            loc_ans_list[i] = ann.ann_id
         else
            assert(false, 'undefined answer type')
         end
         self.batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]
      end
      -- heavy weight fetching (use thread)
      local loc_batch_size = self.batch_size
      local loc_question_file_dir = self.question_file_dir
      local loc_answer_file_dir = self.answer_file_dir
      local loc_ans_type = self.ans_type
      self.pool:addjob(
         function ()
            local loc_image = require 'image'
            local pre_img = torch.zeros(loc_batch_size, 3, height, width)
            local pre_batch_qwords = {}
            local pre_batch_awords = {}
            for i = 1, loc_batch_size do
               local img_path = paths.concat(string.format('%s/%s',
                                                    cocoimgpath,loc_subtype_list[i]),
                                             loc_img_list[i]) 
               local img = loc_image.load(img_path)
               if randcrop then
                  img = loc_image.scale(img, rewidth, reheight)
               else
                  img = loc_image.scale(img, width, height)
               end
               if img:size()[1] == 1 then
                  img = img:repeatTensor(3,1,1)
               end
               if randcrop then
                  local cx1 = torch.random() % (rewidth-width) + 1
                  local cy1 = torch.random() % (reheight-height) + 1
                  local cx2 = cx1 + width
                  local cy2 = cy1 + height
                  img = image.crop(img, cx1, cy1, cx2, cy2)
               end
               img = img:index(1, torch.LongTensor{3,2,1})
               img = img * 255 - mean_bgr
               img = img:contiguous()
               pre_img[i] = img
               -- question
               local q_file_path = string.format('%s/%s_%d.t7', 
                                       string.format(loc_question_file_dir,loc_subtype_list[i]),
                                       loc_subtype_list[i], loc_qid_list[i])
               local question = torch.load(q_file_path)
               pre_batch_qwords[i] = question.tokenized
               -- answer
               if loc_ans_type == 'major' then
                  pre_batch_awords[i] = loc_ans_list[i]
               elseif loc_ans_type == 'every' then
                  local answer_file_path = string.format('%s/%s_%d.t7',
                                              string.format(loc_answer_file_dir,loc_subtype_list[i]),
                                              loc_subtype_list[i],loc_ans_list[i])
                  local answers = torch.load(answer_file_path)
                  local rand_n = (torch.random() % (#answers)) + 1
                  pre_batch_awords[i] = answers[rand_n].answer
               else
                  assert(false, 'undefined answer type')
               end
            end
            return pre_img, pre_batch_qwords, pre_batch_awords
         end,
         function (pre_img, pre_batch_qwords, pre_batch_awords)
            self.batch_img = pre_img
            self.batch_qwords = pre_batch_qwords
            self.batch_awords = pre_batch_awords
         end
      )
   else
      local batch_q = torch.zeros(self.batch_size, self.seq_len)
      local batch_q_len = torch.zeros(self.batch_size)
      local batch_a = torch.zeros(self.batch_size)
      local batch_img = torch.zeros(self.batch_size, 3, height, width)
      for i = 1, self.batch_size do
         local ann_idx = self.batch_order[i + self.batch_index]
         local ann = self.annotations[ann_idx]
         local cocoimg_name = string.format('COCO_%s_%012d.jpg',
                                     ann.data_subtype, ann.image_id)
         local img_path = paths.concat(string.format('%s/%s',
                                              cocoimgpath,ann.data_subtype),
                                       loc_img_list[i])
         local img = image.load(img_path)
         if randcrop then
            img = image.scale(img, rewidth, reheight)
         else
            img = image.scale(img, width, height)
         end
         if img:size()[1] == 1 then
            img = img:repeatTensor(3,1,1)
         end
         if randcrop then
            local cx1 = torch.random() % (rewidth-width) + 1
            local cy1 = torch.random() % (reheight-height) + 1
            local cx2 = cx1 + width
            local cy2 = cy1 + height
            img = image.crop(img, cx1, cy1, cx2, cy2)
         end
         img = img:index(1, torch.LongTensor{3,2,1})
         img = img * 255 - mean_bgr
         img = img:contiguous()
         batch_img[i] = img
         batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]

         local q_file_path = string.format('%s/%s_%d.t7',
                                 string.format(self.question_file_dir,ann.data_subtype),
                                 ann.data_subtype, ann.question_id)
         local question = torch.load(q_file_path)
         for k, v in pairs(question.tokenized) do
            local word = string.lower(v)
            if vocab_map[word] == nil then
               batch_q[i][k] = vocab_map['<empty>']
            else
               batch_q[i][k] = vocab_map[word]
            end
         end
         if self.ans_type == 'major' then
            local ans_word = string.lower(ann.multiple_choice_answer)
            if ans_map[ans_word] == nil then
               batch_a[i] = ans_map['<empty>']
            else
               batch_a[i] = ans_map[ans_word]
            end
         elseif self.ans_type == 'every' then
            local answer_file_path = string.format('%s/%s_%d.t7',
                                         string.format(self.answer_file_dir,ann.data_subtype),
                                         ann.data_subtype,ann.ann_id)
            local answers = torch.load(answer_file_path)
            local rand_n = (torch.random() % (#answers)) + 1
            local ans_word = string.lower(answers[rand_n].answer)
            if ans_map[ans_word] == nil then
               batch_a[i] = ans_map['<empty>']
            else
               batch_a[i] = ans_map[ans_word]
            end
         else
            assert(false, 'undefined answer type')
         end
      end

      self.batch_index = self.batch_index + self.batch_size
      if (self.batch_index + self.batch_size) > self.ex_num_train then
         self:reorder()
      end
   end
   return batch_img:clone(), batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone()
end
function dataclass:next_batch(vocab_map, ans_map)
   local batch_q = torch.zeros(self.batch_size, self.seq_len)
   local batch_q_len = torch.zeros(self.batch_size)
   local batch_a = torch.zeros(self.batch_size)
   for i = 1, self.batch_size do
      local ann_idx = self.batch_order[i + self.batch_index]
      local ann = self.annotations[ann_idx]
      local q_file_path = string.format('%s/%s_%d.t7', 
                              string.format(self.question_file_dir,ann.data_subtype),
                              ann.data_subtype, ann.question_id)
      local question = torch.load(q_file_path)
      for k, v in pairs(question.tokenized) do
         local word = string.lower(v)
         if vocab_map[word] == nil then
            batch_q[i][k] = vocab_map['<empty>']
         else
            batch_q[i][k] = vocab_map[word] 
         end
      end
      batch_q_len[i] = self.question_len[self.batch_order[i+self.batch_index]]

      if self.ans_type == 'major' then
         local ans_word = string.lower(self.annotations[ann_idx].multiple_choice_answer)
         if ans_map[ans_word] == nil then
            batch_a[i] = ans_map['<empty>']
         else
            batch_a[i] = ans_map[ans_word]
         end
      elseif self.ans_type == 'every' then
         local answer_file_path = string.format('%s/%s_%d.t7',
                                      string.format(self.answer_file_dir,ann.data_subtype),
                                      ann.data_subtype,ann.ann_id)
         local answers = torch.load(answer_file_path)
         local rand_n = (torch.random() % (#answers)) + 1 
         local ans_word = string.lower(answers[rand_n].answer)
         if ans_map[ans_word] == nil then
            batch_a[i] = ans_map['<empty>']
         else
            batch_a[i] = ans_map[ans_word]
         end
      else
         assert(false, 'undefined answer type')
      end
   end
   self.batch_index = self.batch_index + self.batch_size
   if (self.batch_index + self.batch_size) > self.ex_num_train then
      self:reorder()
   end

   return batch_q:transpose(1,2):clone(), batch_q_len:clone(), batch_a:clone()
end
function dataclass:set_batch_order_option(opt_batch_order)
   if opt_batch_order == 1 then
      print(string.format('[%s] set batch order option 1 : shuffle', self.data_subtype))
   elseif opt_batch_order == 2 then
      print(string.format('[%s] set batch order option 2 : inorder', self.data_subtype))
   elseif opt_batch_order == 3 then
      print(string.format('[%s] set batch order option 3 : sort', self.data_subtype))
   elseif opt_batch_order == 4 then
      print(string.format('[%s] set batch order option 4 : randsort', self.data_subtype))
   else
      assert(true, 'set_opt_batch_order error: this batch order option is not yet defined')
   end
   self.opt_batch_order = opt_batch_order
end
function dataclass:reorder()
   if self.opt_batch_order == 1 then
      self:shuffle()
   elseif self.opt_batch_order == 2 then
      self:inorder()
   elseif self.opt_batch_order == 3 then
      self:sort()
   elseif self.opt_batch_order == 4 then
      self:randsort()
   else
      assert(true, 'reorder error: this batch order option is not yet defined')
   end
end
function dataclass:inorder()
   -- in order
   self.batch_index = 0
   self.batch_order = torch.range(1, self.ex_num_train)
   self.prefetch_init = false
end
function dataclass:shuffle()
   -- random order
   self.batch_index = 0
   self.batch_order = torch.randperm(self.ex_num_train)
   self.prefetch_init = false
end
function dataclass:randsort()
   -- sort according to sequence lenth, but if sequence lengths are equal, shuffle
   self.prefetch_init = false
   self.batch_index = 0
   local sorted, loc_batch_order = self.question_len:sort()
   self.batch_order = loc_batch_order:clone()
   local i = 1
   while i < sorted:nElement()-1 do
      local i_start = i
      local i_end
      for j = i_start, sorted:nElement() do
         if sorted[j] > sorted[i_start] then
            i_end = j
            break
         end
         if j == sorted:nElement() then
            i_end = j+1
         end
      end
      local rand_order = torch.randperm(i_end - i_start)
      for k = 1, i_end-i_start do
         self.batch_order[i_start+k-1] = loc_batch_order[rand_order[k]+i_start-1]
      end
      i = i_end
   end
end
function dataclass:sort()
   self.prefetch_init = false
   self.batch_index = 0
   sorted, self.batch_order = self.question_len:sort()
end
function dataclass:reset_batch_pointer()
   self.batch_index = 0
end
function vqa_loader.load_data(vqa_dir, taskType, ansType, seq_len, batch_size, 
                              opt_prefetch, opt_split, test_batch_size, add_ans2vocab, cache_dir)
   local vqa_data = {}
   setmetatable(vqa_data, vqa_loader)

   local ann_dir = vqa_dir .. '/Annotations'
   local question_dir = vqa_dir .. '/Questions'
   
   vqa_data.vocab_size = 0
   vqa_data.vocab_map = {}
   vqa_data.vocab_dict = {}
   vqa_data.answer_size = 0
   vqa_data.answer_map = {}
   vqa_data.answer_dict = {}
   vqa_data.max_sentence_len = 0
   vqa_data.seq_len = seq_len
   opt_prefetch = opt_prefetch or false
   opt_split = opt_split or 'trainval'
   test_batch_size = test_batch_size or batch_size
   add_ans2vocab = add_ans2vocab or false

   local cache_path = paths.concat(cache_dir, 
                            string.format('vqa_data_cache_%s_%s_%d', ansType, opt_split, seq_len))
   
   print(string.format('training data cache path is..: %s', cache_path))
   local is_cached = path.isfile(cache_path) 
   if is_cached then
      print(string.format('cache exists.. load cache'));
   else
      print(string.format('cache doesnt exist'));
   end

   if opt_split == 'trainval' then
      vqa_data.train_data = dataclass.load_data({'train2014'}, 
                                             vqa_data, vqa_dir, taskType, ansType,
                                             seq_len, batch_size, opt_prefetch, true, add_ans2vocab, 
                                             is_cached)
      vqa_data.val_data = dataclass.load_data({'val2014'}, 
                                             vqa_data, vqa_dir, taskType, ansType,
                                             seq_len, test_batch_size, opt_prefetch, false, false,
                                             is_cached)
   elseif opt_split == 'test2015' then
      vqa_data.train_data = dataclass.load_data({'train2014', 'val2014'}, 
                                             vqa_data, vqa_dir, taskType, ansType,
                                             seq_len, batch_size, opt_prefetch, true, add_ans2vocab,
                                             is_cached)
      vqa_data.test_data = dataclass.load_testdata('test2015',
                                             vqa_data, vqa_dir, taskType,
                                             seq_len, test_batch_size, opt_prefetch,
                                             is_cached)
   elseif opt_split == 'test-dev2015' then
      vqa_data.train_data = dataclass.load_data({'train2014', 'val2014'}, 
                                             vqa_data, vqa_dir, taskType, ansType,
                                             seq_len, batch_size, opt_prefetch, true, add_ans2vocab,
                                             is_cached)
      vqa_data.test_data = dataclass.load_testdata('test-dev2015',
                                             vqa_data, vqa_dir, taskType,
                                             seq_len, test_batch_size, opt_prefetch,
                                             is_cached)
   elseif opt_split == 'all' then
      vqa_data.train_data = dataclass.load_data({'train2014'}, 
                                             vqa_data, vqa_dir, taskType, ansType,
                                             seq_len, batch_size, opt_prefetch, true, add_ans2vocab,
                                             is_cached)
      vqa_data.val_data = dataclass.load_data({'val2014'}, 
                                             vqa_data, vqa_dir, taskType, ansType,
                                             seq_len, batch_size, opt_prefetch, false, false,
                                             is_cached)
      vqa_data.test_data = dataclass.load_testdata('test2015',
                                             vqa_data, vqa_dir, taskType,
                                             seq_len, test_batch_size, opt_prefetch,
                                             is_cached)
      vqa_data.testdev_data = dataclass.load_testdata('test-dev2015',
                                             vqa_data, vqa_dir, taskType,
                                             seq_len, test_batch_size, opt_prefetch,
                                             is_cached)
   else
      assert(false, 'undefined split option')
   end

   if is_cached then
      print(string.format('loading cache..: %s', cache_path))
      local cache = torch.load(cache_path)
      vqa_data.vocab_map = cache.vocab_map
      vqa_data.vocab_size = cache.vocab_size
      vqa_data.vocab_dict = cache.vocab_dict
      vqa_data.max_sentence_len = cache.max_sentence_len
      vqa_data.answer_map = cache.answer_map
      vqa_data.answer_dict = cache.answer_dict
      vqa_data.answer_size = cache.answer_size
      vqa_data.max_ans_len = cache.max_ans_len
      if opt_split == 'trainval' then
         vqa_data.train_data.question_len = cache.train_question_len
         vqa_data.val_data.question_len = cache.val_question_len
      elseif opt_split == 'test2015' or opt_split == 'test-dev2015' then
         vqa_data.train_data.question_len = cache.train_question_len
         vqa_data.test_data.question_len = cache.test_question_len
         vqa_data.test_data.max_sentence_len = cache.test_max_sentence_len
      elseif opt_split == 'all' then
         vqa_data.train_data.question_len = cache.train_question_len
         vqa_data.val_data.question_len = cache.val_question_len
         vqa_data.test_data.question_len = cache.test_question_len
         vqa_data.test_data.max_sentence_len = cache.test_max_sentence_len
         vqa_data.testdev_data.question_len = cache.testdev_question_len
         vqa_data.testdev_data.max_sentence_len = cache.testdev_max_sentence_len
      end
      print('done')
   end
 
   -- cache
   if is_cached == false and path.isdir(cache_dir) then
      print(string.format('saving cache..: %s', cache_path))
      local cache = {} 
      cache.vocab_map = vqa_data.vocab_map
      cache.vocab_size = vqa_data.vocab_size
      cache.vocab_dict = vqa_data.vocab_dict
      cache.max_sentence_len = vqa_data.max_sentence_len
      cache.answer_map = vqa_data.answer_map
      cache.answer_dict = vqa_data.answer_dict
      cache.answer_size = vqa_data.answer_size
      cache.max_ans_len = vqa_data.max_ans_len
      if opt_split == 'trainval' then
         cache.train_question_len = vqa_data.train_data.question_len
         cache.val_question_len = vqa_data.val_data.question_len
      elseif opt_split == 'test2015' or opt_split == 'test-dev2015' then
         cache.train_question_len = vqa_data.train_data.question_len
         cache.test_question_len = vqa_data.test_data.question_len
         cache.test_max_sentence_len = vqa_data.test_data.max_sentence_len
      elseif opt_split == 'all' then
         cache.train_question_len = vqa_data.train_data.question_len
         cache.val_question_len = vqa_data.val_data.question_len
         cache.test_question_len = vqa_data.test_data.question_len
         cache.test_max_sentence_len = vqa_data.test_data.max_sentence_len
         cache.testdev_question_len = vqa_data.testdev_data.question_len
         cache.testdev_max_sentence_len = vqa_data.testdev_data.max_sentence_len
      end
      torch.save(cache_path, cache)
      print('done')
   end

   return vqa_data
end

return vqa_loader
