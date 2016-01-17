th porting_VQA_data.lua -dataSubType train2014 -target annotation
th porting_VQA_data.lua -dataSubType train2014 -target question -taskType OpenEnded
th porting_VQA_data.lua -dataSubType train2014 -target question -taskType MultipleChoice
th porting_VQA_data.lua -dataSubType val2014 -target annotation
th porting_VQA_data.lua -dataSubType val2014 -target question -taskType OpenEnded
th porting_VQA_data.lua -dataSubType val2014 -target question -taskType MultipleChoice
th porting_VQA_data.lua -dataSubType val2014 -target question -taskType OpenEnded
th porting_VQA_data.lua -dataSubType val2014 -target question -taskType MultipleChoice
th porting_VQA_data.lua -dataSubType test2015 -target question -taskType OpenEnded
th porting_VQA_data.lua -dataSubType test2015 -target question -taskType MultipleChoice
th porting_VQA_data.lua -dataSubType 'test-dev2015' -target question -taskType OpenEnded
th porting_VQA_data.lua -dataSubType 'test-dev2015' -target question -taskType MultipleChoice
