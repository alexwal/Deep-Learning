local high_level = require 'high_level_convnet.lua'

-- SCRIPT to run training/evaluation using high level functions of Torch.

-- Loading CIFAR 10 data
if not high_level.file_exists("cifar10torchsmall.zip") then
  if not high_level.file_exists("cifar10-train.t7") or not high_level.file_exists("cifar10-test.t7") then
    os.execute("wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip")
    os.execute("unzip cifar10torchsmall.zip")
  end
end
local train_set = torch.load("cifar10-train.t7") -- in format: dataset = {data = ..., label = ...}
local test_set = torch.load("cifar10-test.t7")
local classes = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}

high_level.format_dataset(train_set, "train")
high_level.format_dataset(test_set, "test")
high_level.preprocess(train_set, test_set)
local net = high_level.create_model()
print("Lenet5:")
print(net)
local criterion = nn.ClassNLLCriterion() -- loss function
high_level.train(train_set, net, criterion)
high_level.evaluate(train_set, net, criterion, classes)
high_level.evaluate(test_set, net, criterion, classes)

