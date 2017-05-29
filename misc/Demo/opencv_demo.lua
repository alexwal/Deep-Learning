-- Author: Alex Walczak, 2016
local cv = require 'cv'
require 'cv.highgui'
require 'cv.videoio'
require 'cv.imgproc'
require 'image'
require 'nn'
local utils = require 'utils'
require 'Head'

print '==> Running webcam demo with additional evolving fully-connected layer'
local cmd = torch.CmdLine()
cmd:option('-rate',0.02,'learningRate')
--cmd:option('-rate',0.005,'learningRate')
local params = cmd:parse(arg)
print('Command line options:')
print(params)

local cap = cv.VideoCapture{device=0}
    if not cap:isOpened() then
    print("Failed to open the default camera")
    os.exit(-1)
end

print '==> loading decapitated model'
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

-- keep a running average, std of images seen
local CNN = torch.load('./inceptionv3.net.decap.CPU')
local classes = {}
local new_feature_vectors = {}

function add_feature_vector(fv, target_class)
    print('training class:', target_class)
    if head:seen_class(target_class) == false then
        new_feature_vectors[target_class] = {}
        head:add_class(target_class)
        classes[target_class] = 'New class:' .. tostring(target_class)
    end
    table.insert(new_feature_vectors[target_class], fv)
end

--- test + train functions

-- Camera handling
cv.namedWindow{winname="Inception V3 Classification Demo", flags=cv.WINDOW_AUTOSIZE}
local _, frame = cap:read{}
local M = 299 -- new image dimension 

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

local function train()
    head:evaluate()
    while true do
       local w, h = frame:size(2), frame:size(1)
       local n = math.min(w,h)
       local crop = cv.getRectSubPix{image=frame, patchSize={n,n}, center={w/2, h/2}}
       I = cv.resize{src=crop, dsize={M,M}}:permute(3,1,2):clone():float() -- channels to RGB
       local feature_vector = CNN:forward(I:view(1,3,M,M)):squeeze() -- computation on 'I' image, net expects RGB image
       if #classes >= 1 then
           if #classes == 1 and tostring(criterion) == 'nn.CrossEntropyCriterion' then
              cv.putText{
                 img=crop,
                 text = classes[1] .. ': + 1.00',
                 org={10, 35},
                 fontFace=cv.FONT_HERSHEY_DUPLEX,
                 fontScale=1,
                 color={255, 255, 255},
                 thickness=1
              }
           else
               local scores, class_idx = head:test(feature_vector:double())
               for i=1,math.min(8,#classes) do
                  cv.putText{
                     img=crop,
                     text = classes[class_idx[i]] .. ': ' .. tostring(scores[i]),
                     org={10,10 + i * 25},
                     fontFace=cv.FONT_HERSHEY_DUPLEX,
                     fontScale=1,
                     color={0, 0, 0},
                     thickness=1
                  }
               end
           end
       end
       cv.imshow{winname="Inception V3 Classification Demo", image=crop} --shows 'crop' image (channels BGR for ocv)

       -- Handling key presses
       local key = cv.waitKey{1}
       if key == 27 then -- 'ESC' to exit
           do return end
       elseif 49 <= key and key <= 56 then--57 then --'1' - '9' to train 9 classes (in order)
           local training_class = key - 49 + 1
           add_feature_vector(feature_vector:double(), training_class)
           single_training_epoch()
       end
       cap:read{image=frame}
    end
end

-- Start training
train()
