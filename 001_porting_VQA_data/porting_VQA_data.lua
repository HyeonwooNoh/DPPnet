local stringx = require 'pl.stringx'
local cjson = require 'cjson'

cmd = torch.CmdLine()
cmd:text()
cmd:text('porting VQA data')
cmd:text()
cmd:option('-vqa_dir', './data/VisualQA/VQA', 'directory for original VQA dataset')
cmd:option('-taskType', 'OpenEnded', 'task type: [OpenEnded|MultipleChoice]')
cmd:option('-dataSubType', 'train2014', 'data sub type: [train2014|val2014]')
cmd:option('-save_dir', './data/VQA_torch', 'directory for saving ported VQA data')
cmd:option('-target', 'annotation', 'target file [annotation|question]')

opt = cmd:parse(arg or {})

if opt.target == 'annotation' then
   local ann_file_path = paths.concat(string.format('%s/Annotations',opt.vqa_dir),
                         string.format('mscoco_%s_annotations.json',opt.dataSubType))

   local ann_file = io.open(ann_file_path, 'r')
   local ann_string = ann_file:read()
   local ann_json = cjson.decode(ann_string)
   ann_file:close()
   local annotations = {}

   local save_ann_dir = string.format('%s/Annotations/mscoco_%s_annotations',
                                       opt.save_dir,opt.dataSubType)
   local save_answer_dir = string.format('%s/answers', save_ann_dir)
   -- create directory
   os.execute(string.format('mkdir -p %s', save_answer_dir)) 
   local num_ann = #ann_json.annotations 
   local i = 1
   for k, v in pairs(ann_json.annotations) do
      print(string.format('[Annotation|%s]%d/%d',opt.dataSubType, k, num_ann))
      local answer_file_path = string.format('%s/%s_%d.t7', save_answer_dir,opt.dataSubType, k)
      torch.save(answer_file_path, v.answers)

      annotations[k] = {}
      annotations[k].question_id = v.question_id
      annotations[k].multiple_choice_answer = v.multiple_choice_answer
      annotations[k].answer_type = v.answer_type
      annotations[k].image_id = v.image_id
      annotations[k].ann_id = k
      annotations[k].question_type = v.question_type

      if i % 1000 == 0 then
         collectgarbage()
      end
      i = i + 1
   end
   local save_ann_path = string.format('%s/annotations.t7', save_ann_dir)
   torch.save(save_ann_path, annotations)
elseif opt.target == 'question' then
   local question_file_path = paths.concat(string.format('%s/Questions',opt.vqa_dir), 
                      string.format('%s_mscoco_%s_questions.json',opt.taskType,opt.dataSubType))

   local question_file = io.open(question_file_path, 'r')
   local question_string = question_file:read()
   local question_json = cjson.decode(question_string)
   question_file:close() 

   local save_question_dir = string.format('%s/Questions/%s_mscoco_%s_questions', 
                                           opt.save_dir, opt.taskType, opt.dataSubType)
   local question_list = {}
   -- create directory
   os.execute(string.format('mkdir -p %s', save_question_dir))
   local i = 1
   for k, v in pairs(question_json.questions) do
      local coco_subtype = opt.dataSubType
      if opt.dataSubType == 'test2015' or opt.dataSubType == 'test-dev2015' then
         coco_subtype = 'test2015'
      end
      print(string.format('[Question|%s|%s|%s]%d', opt.dataSubType,coco_subtype,opt.taskType,i))
      local question_file_path = string.format('%s/%s_%d.t7', 
                                 save_question_dir,coco_subtype,v.question_id)
      local question = {}
      question.question_id = v.question_id
      question.image_id = v.image_id
      question.question = v.question
      local q_ids = {}
      q_ids.image_id = v.image_id
      q_ids.question_id = v.question_id
      table.insert(question_list,q_ids)

      local q_sents = v.question
      q_sents = string.sub(q_sents,1,string.len(q_sents)-1)
      local tokenized = stringx.split(q_sents)

      question.tokenized = tokenized
      if opt.taskType == 'MultipleChoice' then
         question.multiple_choices = v.multiple_choices
      end

      torch.save(question_file_path, question)
      if i % 1000 == 0 then
         collectgarbage()
      end
      i = i + 1
   end
   local save_question_list_path = string.format('%s/question_list.t7', save_question_dir)
   torch.save(save_question_list_path, question_list)
else
   assert(false, 'undefined target value')
end 
















