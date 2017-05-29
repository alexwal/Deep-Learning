require 'nn'
local high_level = {}

-- Utililty:
function high_level.file_exists(name)
  local f = io.open(name, "r")
  if f ~= nil then
    io.close(f)
    return true
  else
    return false 
  end
end

--- // --- // --- // --- // --- // --- // --- // --- // 
-- // --- // --- // --- // --- // --- // --- // --- // 

-- MODEL
function high_level.create_model()
  local model = nn.Sequential()
  -- nn.SpatialConvolution(nInputPlane, nOutputPlane, 
  --                                kW,           kH, -- kernel width, height
  --                            [dW=1],       [dH=1], -- stride of filter along W (cols), H (rows)
  --                          [padW=0],     [padH=0]) -- zero-padding of input (pad = (k-1)/2 for `same` size output)
  model:add(nn.SpatialConvolution(3, 6, 5, 5)) -- add spatial convolution layer 
  model:add(nn.ReLU()) -- add ReLU layer
  -- SpatialMaxPooling(kW, kH, [ dW, dH, padW, padH])
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialConvolution(6, 16, 5, 5))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  -- Reshape Tensor of size 16*5*5 (x1?) into 1D vector with length 16*5*5
  model:add(nn.View(16*5*5))
  -- Fully connected layer
  model:add(nn.Linear(16*5*5, 120))
  model:add(nn.ReLU())
  model:add(nn.Linear(120, 84))
  model:add(nn.ReLU())
  model:add(nn.Linear(84, 10))
  model:add(nn.LogSoftMax())
  return model
end

-- FORMATTING
function high_level.format_dataset(dataset, name)
  -- Torch supports Stochastic Gradient 
  -- To use it, the training data must have :size() function high_level.and 
  -- index operator ([index])
  -- add index operator to the training set
  -- Input: dataset.data/.label indexed contains data/labels.
  metatable = {__index = function(t, i) return {t.data[i], t.label[i]} end}
  setmetatable(dataset, metatable)
  dataset.data = dataset.data:double() -- convert from byte to double type
  function dataset:size()
    return self.data:size(1)
  end
  dataset.name = name
end

-- PREPROCESSING
function high_level.preprocess(train_set, test_set)
  -- subtract mean and normalize channels
  local mean, stdv = {}, {}
  for i = 1, 3 do 
    -- 1. train data:
    mean[i] = train_set.data[ {{}, {i}, {}, {}} ]:mean() -- mean estimation (sample mean)
    train_set.data[{{}, {i}, {}, {}}]:add(-mean[i]) -- mean subtraction
    stdv[i] = train_set.data[{{}, {i}, {}, {}}]:std() -- std estimation
    train_set.data[{ {}, {i}, {}, {} }]:div(stdv[i]) -- std scaling
    -- 2. test data:
    test_set.data[{{}, {i}, {}, {}}]:add(-mean[i])
    test_set.data[{{}, {i}, {}, {}}]:div(stdv[i])
  end
  train_set.mean = mean
  train_set.stdv = stdv
end

-- TRAINING 
function high_level.train(dataset, model, criterion)
  -- zero the internal gradients
  model:zeroGradParameters()
  model:training()
  local trainer = nn.StochasticGradient(model, criterion)
  trainer.learningRate = 0.001
  trainer.maxIteration = 1
  trainer:train(dataset)
  return trainer
end

-- TESTING / EVALUATION
function high_level.evaluate(dataset, model, criterion, classes)
  local name = dataset.name or ""
  print("\nEvaluating " .. name .. "...")
  model:evaluate()

  -- Example of inference on one sample.
  print("\nResult of inference on one sample:")
  local sample, label = unpack(dataset[100])
  local predicted = model:forward(sample) 
  predicted:exp() -- convert LogSoftMax --> SoftMax == a probability disribution over classes
  print("Label:", classes[label])
  for i = 1, predicted:size(1) do
    print(classes[i], predicted[i]) 
  end

  -- Compute accuracy on dataset.
  local class_correct = torch.totable(torch.zeros(#classes)) 
  local class_totals = torch.totable(torch.zeros(#classes)) 
  local global_correct = 0
  local loss = 0
  for i = 1, dataset:size() do
    local sample, groundtruth = unpack(dataset[i])
    local prediction = model:forward(sample)
    local scores, indexes = torch.sort(prediction, true) -- reversed (decreasing) = true
    if (groundtruth == indexes[1]) then
      class_correct[groundtruth] = class_correct[groundtruth] + 1
      global_correct = global_correct + 1
    end
    class_totals[groundtruth] = class_totals[groundtruth] + 1
    loss = loss + criterion:forward(prediction, groundtruth)
  end

  print("\nDistribution of classes in " .. name .. " set:")
  for i = 1, #classes do
    print(classes[i], (class_totals[i] / dataset:size()) * 100 .. " %")
  end

  print("\nClass-level accuracy:")
  for i = 1, #classes do
    print(classes[i], (class_correct[i] / class_totals[i]) * 100 .. " %")
  end

  local accuracy = global_correct / dataset:size()
  print("\nGlobal accuracy:", accuracy * 100 .. " %")
  print("\nGlobal average loss:", loss / dataset:size()) 
end

--- // --- // --- // --- // --- // --- // --- // --- // 
-- // --- // --- // --- // --- // --- // --- // --- // 

return high_level

