local qa_utils = {}

qa_utils.question_type = { [0] = 'object',
                           [1] = 'number',
                           [2] = 'color',
                           [3] = 'location'}

function qa_utils.cocofile_name(imgset, id)
   return string.format('COCO_%s_%012d', imgset, id)
end
function qa_utils.cocoimg_name(imgset, id)
   return string.format('COCO_%s_%012d.jpg', imgset, id)
end
function qa_utils.cocofeat_name(imgset, id)
   return string.format('COCO_%s_%012d.t7', imgset, id)
end

return qa_utils
