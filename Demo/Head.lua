-- A Head is an adaptable additional architecture that is appended
-- to a decaptitated network (see: Guillotine.lua).
require 'image'
require 'xlua'
require 'nn'

local Head, Parent = torch.class('nn.Head', 'nn.Sequential')

function Head:__init(input_dimension, criterion, learningRate)
   Parent.__init(self)
   self.indim = input_dimension
   self.seen_classes = {} -- converts raw to target
   self.seen_classes_num = 0
   self.criterion = criterion
   self.learningRate = learningRate
end

function Head:target_class(raw_class)
    -- Because the confusion matrix expects classes
    -- to be numbered from 1 to #classes, self.seen_classes
    -- converts the "raw" class label into the properly
    -- ordered label from 1 to #classes -- i.e. the "target" class.
    -- (raw_class = 1 if seen first, = 2 if seen second, etc.)
    return self.seen_classes[raw_class]
end

function Head:seen_class(raw_class)
    -- input is assumed to be raw class label
    return self:target_class(raw_class) ~= nil
end

function Head:set_target_class(raw_class, target_class)
    self.seen_classes[raw_class] = target_class
end

function Head:format_label(orig_fv_label) --Future work: choose labels +/- 1 and 0/1
   -- orig_fv_label is the raw class label
   local label
   -- Expects scalar class index label 
   local crit_name = tostring(self.criterion)
   if crit_name == 'nn.MarginCriterion' then
       -- Expects vector +/- 1 label 
       label = -1*torch.ones(self.seen_classes_num)
       local target_class = self:target_class(orig_fv_label)
       label[target_class] = 1
   elseif crit_name == 'nn.CrossEntropyCriterion' then
       -- Expects scalar class index label 
       label = self:target_class(orig_fv_label)
   end
   return label
end

function Head:add_class(new_class)
    -- input is assumed to be raw class label
    assert(not self:seen_class(new_class), 'Class shouldn\'t be added twice.')
    self.seen_classes_num = self.seen_classes_num + 1
    self:set_target_class(new_class, self.seen_classes_num)-- assigns the next in order label
    local seen_classes_num = self.seen_classes_num

    if self:get(1) == nil then
        self:insert(nn.Dropout(0.2), 1)
        self:insert(nn.Linear(self.indim, 1), 2)
    else
        local old_weight = self:get(2).weight:clone()
        local old_bias = self:get(2).bias:clone()
        self:remove(2)
        self:insert(nn.Linear(self.indim, seen_classes_num), 2)
        self:get(2).bias[{{1, seen_classes_num-1}}]:copy(old_bias)
        self:get(2).weight[{{1, seen_classes_num-1}}]:copy(old_weight)
    end
end

function Head:grad_update(x, raw_class, model)
   -- model argument can be an additional Head. This way, you
   -- can stack two Heads, and do a gradient descent update.
   local crit_name = tostring(self.criterion)
   if crit_name == 'nn.CrossEntropyCriterion' then
       if self.seen_classes_num < 2 then
           return nil
       end
   end
   if model == nil then
       self:training()
       local pred = self:forward(x:double()) -- DOUBLE
       local label = self:format_label(raw_class)
       local err = self.criterion:forward(pred, label) 
       local gradCriterion = self.criterion:backward(pred, label)
       self:zeroGradParameters()
       self:backward(x, gradCriterion)
       self:updateParameters(self.learningRate)
       self:evaluate()
   else
       model:training()
       local pred = model:forward(x:double()) -- DOUBLE
       local label = self:format_label(raw_class)
       local err = self.criterion:forward(pred, label) 
       local gradCriterion = self.criterion:backward(pred, label)
       model:zeroGradParameters()
       model:backward(x, gradCriterion)
       model:updateParameters(self.learningRate)
       model:evaluate()
   end
end

function Head:test(x, model)
    -- find most likely class
    local scores
    if model == nil then
        self:evaluate()
        scores = self:forward(x:double()):clone() -- DOUBLE
    else
        model:evaluate()
        scores = model:forward(x:double()):clone() -- DOUBLE
    end
    local sorted_scores,target_cl = scores:sort(true)
    return sorted_scores, target_cl
end

