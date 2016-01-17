local skipthoughts_GRU ={}

function skipthoughts_GRU.create(params)
   -- params should contain
   -- U  [rnn_size, rnn_size * 2]
   -- W  [word_dim, rnn_size * 2]
   -- b  [rnn_size * 2]
   -- Ux [rnn_size, rnn_size]
   -- Wx [word_dim, rnn_size]
   -- bx [rnn_size]
   local word_dim = params.W:size(1)
   local rnn_size = params.Wx:size(2)

   assert(params.U:size(1) == rnn_size, 'U: [rnn_size, rnn_size * 2]')
   assert(params.U:size(2) == rnn_size * 2, 'U: [rnn_size, rnn_size * 2]')
   assert(params.W:size(1) == word_dim, 'W: [word_dim, rnn_size * 2]')
   assert(params.W:size(2) == rnn_size * 2, 'W: [word_dim, rnn_size * 2]')
   assert(params.b:size(1) == rnn_size * 2, 'b: [rnn_size * 2]')
   assert(params.Ux:size(1) == rnn_size, 'Ux: [rnn_size, rnn_size]')
   assert(params.Ux:size(2) == rnn_size, 'Ux: [rnn_size, rnn_size]')
   assert(params.Wx:size(1) == word_dim, 'Wx: [word_dim, rnn_size]')
   assert(params.Wx:size(2) == rnn_size, 'Wx: [word_dim, rnn_size]')
   assert(params.bx:size(1) == rnn_size, 'bx: [rnn_size]')

   -- layers with parameter
   local linear_Ux = nn.Linear_wo_bias(rnn_size, rnn_size)
   local linear_U = nn.Linear_wo_bias(rnn_size, rnn_size * 2)
   local linear_Wx = nn.Linear(word_dim, rnn_size)
   local linear_W = nn.Linear(word_dim, rnn_size*2)

   -- inputs
   local x = nn.Identity()()
   local prev_h = nn.Identity()()
 
   -- gates
   local i2rz = linear_W(x)
   local h2rz = linear_U(prev_h)

   local gates = nn.CAddTable()({i2rz, h2rz})
   local reshaped_gates = nn.Reshape(2, rnn_size)(gates)
   local sliced_gates = nn.SplitTable(2)(reshaped_gates)

   local r_gate = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
   local z_gate = nn.Sigmoid()(nn.SelectTable(2)(sliced_gates))

   -- candidate activation
   local mid_Wx = linear_Wx(x)
   local mid_Uh = linear_Ux(prev_h)
   local mid_Urh = nn.CMulTable()({r_gate, mid_Uh})
   local mid_h = nn.Tanh()(nn.CAddTable()({mid_Wx, mid_Urh}))

   -- next hidden state
   local inv_z_gate = nn.AddConstant(1,false)(nn.MulConstant(-1,false)(z_gate))
   local next_h = nn.CAddTable()({nn.CMulTable()({z_gate, prev_h}), 
                                  nn.CMulTable()({inv_z_gate, mid_h})})

   -- construct nngraph module
   gru_module = nn.gModule({x, prev_h}, {next_h})
   
   -- set parameters
   linear_Ux.weight:copy(params.Ux:t())
   linear_U.weight:copy(params.U:t())
   linear_Wx.weight:copy(params.Wx:t())
   linear_Wx.bias:copy(params.bx)
   linear_W.weight:copy(params.W:t())
   linear_W.bias:copy(params.b)
   
   return gru_module
end

return skipthoughts_GRU

