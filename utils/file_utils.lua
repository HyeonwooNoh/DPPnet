
local file_utils = {}

function file_utils.text_read(filename)

   local f = io.open(filename, 'r')
   local lines = {}
   repeat
      line = f:read()
      table.insert(lines, line)
   until line == nil
   f:close() 

   return lines
end
function file_utils.write_text(filename, data_table)
   local f = io.open(filename, 'wt')
   for i, k in pairs(data_table) do
      f:write(k .. '\n')
   end
   f:close()
end
return file_utils
