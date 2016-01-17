

===========================================================================
====                               How to Run                          ====
===========================================================================
main.lua trains mnist with feed forward fully connected neural network. Note
that it only runs on CUDA.
1. run "th main.lua"
- This command runs the standard neural network. You should be able to get
  test error around 1.5%
2. run "th main.lua -hash"
- When -hash is present, it runs HashedNets with default setting (e.g.
  compression ratio, xi, hash_bias and etc). You should be able to get test
  error around 1.5% with 1/8 compression
3. run "th main.lua -hash -compression 0.1 -nhu 500 -nhLayers 2"
- This command runs HashedNets with compression ratio = 0.1, the number of
  hidden units = 500 and the number of hidden layers = 2
3. run "th main.lua -help" for more command line options


===========================================================================
====                           How to Compile                          ====
===========================================================================
1. run "cd ./libhashnn"
2. run "./compile.sh"
dependencies: openblas
(only tested in linux)


===========================================================================
====                           Torch Packages                          ====
===========================================================================
Packages you need to install
1. nn
2. cunn
3. optim
4. mnist
5. xxhash

You can use luarocks to install them
e.g. luarocks install optim


===========================================================================
====                           Standard Output                         ====
===========================================================================
The program would output the test, validation and train error at each epoch,
which looks like follows:

---------------Epoch: 50 of 1000
Current Errors: test: 0.0176 | valid: 0.0169 | train: 0.0020
Optima achieved at epoch 43: test: 0.0172, valid: 0.0168

The second line reports the current errors for epoch 50
The third line reports the errors of best model. "Best model" is defined as
the model with lowest validation errors so far. In this example, the lowest
validation error was achieved at 43rd epoch.

===========================================================================
====         Difference Between Training and Real-world Testing        ====
===========================================================================
This is important!
1. For fast training, our code reconstructs the virtual matrix (i.e. full-size
matrix). So during training, the HashLinear.lua doesn't save any memory during
training.
2. For real-world deployment or testing, one CANNOT simply run HashLinear.lua.
The right way is to copy the hashed weights (h_weight, h_bias) from
HashLinear.lua and use the Equation (8) in the HashedNets paper to calculate
the output of each layer, which is a simple for loop. We will roll out the
code for this part soon.
3. For experimental purpose (i.e. use HashedNets as a baseline in your paper),
HashLinear.lua is enough.



