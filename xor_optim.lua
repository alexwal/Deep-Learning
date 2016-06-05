--Learning XOR with Optim module.
--based on: https://github.com/torch/nn/blob/master/doc/training.md
--Alex Walczak

--first, start server to view outputs in web browser.
--th -ldisplay.start 8000 0.0.0.0

--see Display module documentation https://github.com/szym/display 
--for a simple intro on viewing images when code is running on a server.

--to run code:
--'th xor_optim.lua [LR_DECAY_RATIO=1] [GRID_STEP=1] [GRID_RANGE=10]'
--for example, this line shows nice images:
--'th xor_optim.lua 1 .1 25'

require'nn'
torch = require'torch'
image = require'image'
display = require'display'
display.configure{hostname = '0.0.0.0', port = 8000}

--init model attributes
local model = nn.Sequential();  --make a multi-layer perceptron
local inputs = 2; local outputs = 1; local HUs = 20; --parameters
local learnRate = 0.01

--build model structure
model:add(nn.Linear(inputs, HUs))
model:add(nn.Tanh())
model:add(nn.Linear(HUs, outputs))

--define loss
local criterion = nn.MSECriterion()

--when using optim, we'll feed it data in minibatches
--also, we need to later define a fn feval that will 
--output the loss and the deriv of loss wrt weights,
--given cur weights as a param. typically, feval is
--defined inside our loop over batches and so has access
--to cur minibatch data.

--create minibatch

--preallocate memory
local batchSize = 128*4
local batchInputs = torch.Tensor(batchSize, inputs) --128x2
local batchLabels = torch.DoubleTensor(batchSize)

local sigma = 10

--writes into mem
function createBatch() --randomly
    for i=1,batchSize do
        local input = 10 * torch.randn(2) --adjusting the normal distbn
        local label = 1
        if input[1] * input[2] > 0 then --asks: do we have same signs??
            label = -1
        end
        batchInputs[i]:copy(input)
        batchLabels[i] = label
    end
end

--the flattening
--more on optim: it expects the params being optimized (weights + biases)
--AND their gradients to be stored as one-dim tensors. we get a nice 1D
--View into our params and gradParams with a simple call to model:getParameters()
--[which will make the new tensors].

local params, gradParams = model:getParameters()

--optim will optimize the params in these 2 tensors directly, which is fine
--because our model parameters now have simply become Views (many) onto these
--two large one-dimensional tensors.
--(pre-existing references to original model params (weights + biases) will
--no longer point to model weight and bias after this flttening.)

--training

--we'll use optim's SGD alg. to train our model. so, we'll need to provide its LR
--via an optimization state table.
local optimState = {learningRate = learnRate}

--as stated before, we'll define an eval function inside our training loop over
--our batches and use optim.sgd to run the training.

require'optim'
require'xlua'

local count = 1
numEpochs = 500
numBatches = 20 --== (dataset set / batchsize)
total_iters = numEpochs * numBatches * batchSize

for epoch=1,numEpochs do
    for batchNum=1,numBatches do
        --disp progress
        xlua.progress(count, total_iters)
        count = count + batchSize
        --make a new batch
        createBatch()
        
        --feval takes current weigths as input,
        --outputs loss and gradLoss wrt weights.
        --since our model weights and biases are just
        --Views into gradParams, calling model:backward(...)
        --implicitly updates its values.
        local function feval(params)
            gradParams:zero()--zeros accumulation of grads (since sgd)
            
            local outputs = model:forward(batchInputs)--predn
            local loss = criterion:forward(outputs, batchLabels)
            local dloss_doutput = criterion:backward(outputs, batchLabels) --kinda like an extension of our network
            model:backward(batchInputs, dloss_doutput)--finds: dloss_douput * doutput_dweights = dloss_dweights [gradParams]

            return loss, gradParams
        end

        optim.sgd(feval, params, optimState)
    end
    if count % 1000 == 0 then
        print('learning rate:')
        print(optimState.learningRate) 
    end
    optimState.learningRate = optimState.learningRate*(tonumber(arg[1]) or 1)
end    

x = torch.Tensor{
    {0.5, 0.5},
    {-0.5, 0.5},
    {0.5, -0.5},
    {-0.5, -0.5},
    --{20, 200}, --add your random test points here
    --{-.02, 40},
    --{46, -221},
    --{-1234, 344}
}
print('input:')
print(x)
predn = model:forward(x)
print('raw output:')
print(predn)

--create a grid of results and show image

require'gnuplot'

N = tonumber(arg[3]) or 10
print('Grid range (N):')
print(N)
step = tonumber(arg[2]) or 1
_range = torch.range(-N, N, step)
len = #_range:storage()

local Z = torch.Tensor(len, len)
for i=1,len do
    for j=1,len do
        local input = torch.Tensor{_range[i], _range[j]}
        Z[i][len+1-j] = model:forward(input)
    end
end

display.image(Z, {title = 'Raw output'})
display.image(torch.sign(Z), {title = 'Signum(raw output)'})

