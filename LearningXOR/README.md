### Learning XOR with Optim module.
Based on: https://github.com/torch/nn/blob/master/doc/training.md

First, start server to view outputs in web browser.
  th -ldisplay.start 8000 0.0.0.0

See Display module documentation https://github.com/szym/display 
for a simple intro on viewing images when code is running on a server.

### To run:
  'th xor_optim.lua [LR_DECAY_RATIO=1] [GRID_STEP=1] [GRID_RANGE=10]'
This input gives nice images:
  'th xor_optim.lua 1 .1 25'
