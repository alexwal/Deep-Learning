-- Author: Alex Walczak, 2016
os.execute('clear')
require 'Head'
require 'camera'
require 'xlua'
local utils = require 'utils'
if not xlua.require 'qtwidget' then
    xlua.error('\nFailed to load qt: verify that qlua was called.\ni.e., run >>> qlua demo.lua')
end

print '==> Running webcam demo with additional evolving fully-connected layer'
local cmd = torch.CmdLine()
cmd:option('-rate',0.02,'learningRate')
local params = cmd:parse(arg)

print '==> Loading decapitated model'
-- Download if necessary
if not utils.file_exists('./inceptionv3.net.decap.CPU') then
    -- os.execute('scp -i ~/umbocv/aws/CV.pem ubuntu@52.53.202.111:/mnt/inception/network/inceptionv3.net.decap.CPU .')
    os.execute('wget https://drive.google.com/file/d/0B0ZHOzlXSN7Qd3VHM2ttNTJoLTA/view?usp=sharing')
end

-- Creating a new head for the decapitated network
local fv_dim = 2048 
local learningRate = params.rate 
--local criterion = nn.CrossEntropyCriterion()
local margin = 0.1
local criterion = nn.MarginCriterion(margin)
local head = nn.Head(fv_dim, criterion, learningRate)
head:evaluate()

-- Load feature extractor CNN
local CNN = torch.load('./inceptionv3.net.decap.CPU')
local classes = {}
local new_feature_vectors = {}

-- Add current feature fector when shot is acquired 
function add_feature_vector(fv, target_class)
    print('training class:', target_class)
    if head:seen_class(target_class) == false then
        new_feature_vectors[target_class] = {}
        head:add_class(target_class)
        -- Handle user input for new class's name, or press ENTER to skip 
        local class_label 
        repeat
           io.write("Enter a name for this class: ")
           io.flush()
           class_label = io.read()
        until class_label ~= nil
        if class_label == '' then
            classes[target_class] = 'New class:' .. tostring(target_class)
        else
            classes[target_class] = class_label .. ' (' .. tostring(target_class) .. ')'
        end
        print('Continuing demo...')
    end
    table.insert(new_feature_vectors[target_class], fv)
end

-- Camera handling
local cam = image.Camera{}
local frame = cam:forward()
local scale_by = 1.5
local width, height = scale_by*frame:size(3), scale_by*frame:size(2)
local win = qtwidget.newwindow(width, height, 'Demo')
local M = 299 -- new image dimension 
local feature_vector 

-- Test + train functions
local function single_training_epoch()
    local cl_order = torch.randperm(#new_feature_vectors)
    for cl_index=1,#new_feature_vectors do
        local cl = cl_order[cl_index]
        local fv_order = torch.randperm(#new_feature_vectors[cl])
        for fv_index=1,#new_feature_vectors[cl] do 
            local idx = fv_order[fv_index]
            local feature_vector = new_feature_vectors[cl][idx]
            head:grad_update(feature_vector, cl)
        end
    end
end

-- Handling key presses
function key_press(event)
   local key = string.byte(tostring(event))
   if key == nil then return end
   if key == 27 then -- 'ESC' to exit
        cam:stop()
        os.exit() 
   elseif 49 <= key and key <= 57 then -- Press '1' - '9' to train 9 classes (do in order)
       local training_class = key - 49 + 1
       add_feature_vector(feature_vector:double(), training_class)
       single_training_epoch()
   end
end

-- Attach call back to display window
qt.connect(win.listener, 'sigKeyPress(QString,QByteArray,QByteArray)', key_press)

function draw_scores_on_img(img, scores, class_idx)
    local text -- text to be put on img 
    if #classes >= 1 then
        text = '' 
        for i = 1,math.min(#classes,5) do
            local line = classes[class_idx[i]] .. ' ' .. tostring(scores[i]) .. '\n'
            text = text .. line
        end
    else
        text = 'Press "1" to start learning.'
    end
    local with_scores = image.drawText(img, text, 10, 5, {color = {0, 0, 0}, size = 3})
    return with_scores
end

local function train()
    head:evaluate()
    while true do
       local n = math.min(frame:size(2), frame:size(3))
       local crop = image.crop(frame, 'c', n, n) 
       local I = image.scale(crop, M, M):float()
       local frame = image.scale(frame, width, height)
       feature_vector = CNN:forward(I:view(1,3,M,M)):squeeze() 
       local with_scores
       if #classes >= 1 then
           local scores, class_idx = head:test(feature_vector:double())
           with_scores = draw_scores_on_img(frame, scores, class_idx)
       else
           with_scores = draw_scores_on_img(frame)
       end
       image.display{win=win, image=with_scores}
       frame = cam:forward()
       qt.pause()
    end
    cam:stop()
end

-- Start training
print('\nPress ESC or CTRL-C to exit demo.\n')
train()

