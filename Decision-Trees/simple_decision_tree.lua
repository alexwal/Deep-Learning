-- Builds a decision tree from dataset with ID3 Algorithm
-- Data format: samples on different lines with each dimension
--              separated by white space.
-- First line is header.
-- Second line can be data types (if not specified, data types assumed to all be non-continuous).

-- Parse command line arguments
local cmd = torch.CmdLine()
cmd:text('Create a simple decision tree')
cmd:text()
cmd:text('Options')
cmd:option('-data','iris.txt','properly formatted data file, see comments')
cmd:option('-levels',3,'maximum depth of tree')
cmd:option('-bins',3,'maximum degree of split at node for numerical variables')
cmd:option('-testprop',0.1,'proportion of data to withhold and test on')
cmd:option('-trees',11,'number of trees to test')
cmd:option('-bagging',true,'Whether to do feature bagging, i.e., random subspace selection with replacement')
cmd:text()
local params = cmd:parse(arg)

local data_types = {}

-- Read all lines from a file
local function lines_from(file)
  local all_lines = {}
  for line in io.lines(file) do 
    all_lines[#all_lines + 1] = line
  end
  return all_lines
end

-- Split data lines on separator sep
function split(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t,i = {}, 1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        if data_types[i] == 'num' then
            t[i] = tonumber(str)
        else
            t[i] = str
        end
        i = i + 1
    end
    return t
end

-- Import data from file 
local file = params.data
local lines = lines_from(file)
local header = split(lines[1])
local data_starts_at = 2

-- Read types if provided on second line
if lines[2] == 'data types:' then
    data_types = split(lines[3])
    data_starts_at = 4 
end

-- Extract data by parsing each line
-- Create a random test set
local data = {}
local test_data = {}
local inds = torch.randperm(#lines-data_starts_at+1) + data_starts_at-1

local test_proportion = params.testprop
local num_test = torch.floor(test_proportion * inds:size(1))

for i=1,inds:size(1) do
    local idx = inds[{i}]
    local line = split(lines[idx])
    if i <= num_test then
        table.insert(test_data, line) 
    else
        table.insert(data, line) 
    end
end

local max_bins = params.bins
local function make_bins(data, num_bins)
    -- data is 1D torch tensor
    -- for which we find num_bins bins
    -- to split into
    local num_bins = math.min(data:size(1), num_bins)
    if num_bins == 1 then
        return {torch.median(data):squeeze()}
    end
    local bins, bin_values = {}, {}
    local sorted = torch.sort(data)
    table.insert(bins, sorted[1]) -- insert min data value into bin
    bin_values[sorted[1]] = true

    local step_size = torch.floor(data:size(1)/num_bins)
    -- skip last bin so that no data is left unaccounted
    for bin=step_size,data:size(1)-step_size,step_size do
        if not bin_values[sorted[bin]] then
            table.insert(bins, sorted[bin])
            bin_values[sorted[bin]] = true
        end
    end
    return bins
end

local function preprocess(data,test)
    -- handles num datatypes (numerical, non-categorical attributes)
    data_dims = #data[1] - 1
    for attr_index=1,data_dims do
        if data_types[attr_index] == 'num' then
            local attr_values = {}
            for _,x in ipairs(data) do
                table.insert(attr_values, x[attr_index])
            end
            local bins = make_bins(torch.Tensor(attr_values), max_bins)
            for _,x in ipairs(data) do
                local value = x[attr_index]
                for i=1,#bins-1 do
                    local bin_min, bin_max = bins[i], bins[i+1]
                    if bin_min <= value and value < bin_max then
                        x[attr_index] = tostring(bin_min) .. '-' .. tostring(bin_max)
                    end
                end
                if value < bins[1] then
                    x[attr_index] = '<' .. tostring(bins[1])
                end
                if value >= bins[#bins] then
                    x[attr_index] = '>=' .. tostring(bins[#bins])
                end
            end

            -- set test data values according to training data
            for _,x in ipairs(test) do
                local value = x[attr_index]
                for i=1,#bins-1 do
                    local bin_min, bin_max = bins[i], bins[i+1]
                    if bin_min <= value and value < bin_max then
                        x[attr_index] = tostring(bin_min) .. '-' .. tostring(bin_max)
                    end
                end
                if value < bins[1] then
                    x[attr_index] = '<' .. tostring(bins[1])
                end
                if value >= bins[#bins] then
                    x[attr_index] = '>=' .. tostring(bins[#bins])
                end
            end
        end
    end
end

preprocess(data,test_data)
--print(test_data); do return end

local log2 = function(x) return math.log(x)/math.log(2) end

function count_classes(S)
  -- the last index of each sample x in S
  -- is the sample's class.
  local classes, classes_count = {}, {}
  for _,x in ipairs(S) do
    local class_of_x = x[#x]
    if not classes_count[class_of_x] then
        classes_count[class_of_x] = 1
        table.insert(classes, class_of_x)
    else 
        classes_count[class_of_x] = classes_count[class_of_x] + 1
    end
  end
  return classes, classes_count
end

local function entropy(S)
  -- Compute entropy of a set S: H(S) = - sum_{for all x in S} p(x)*log2(p(x))

  -- Start by finding the proportion of each class in split S
  local classes, classes_count = count_classes(S)

  -- Compute sum 
  local entropy_sum = 0
  for _,cl in ipairs(classes) do
    -- proportion of class cl samples
    local prop_cl = classes_count[cl]/#S
    entropy_sum = prop_cl*log2(prop_cl) + entropy_sum
  end 
  entropy_sum = -entropy_sum
  return entropy_sum
end

-- sample format: x[attr_index] = attr_value
-- x[#x] = class_of_x
local function information_gain(S, attr_index)
    local entropy_S = entropy(S)

    -- We'll group points in S by attr_index's value
    local group_by_attr = {}
   
    -- find all values that this attribute takes on 
    local all_attrs = {num = 0}
    for _,x in ipairs(S) do
        local attr_value = x[attr_index]

        -- all samples x with attribute value attr_value are placed in a single group
        -- that group's index is recorded at all_attrs[attr_value]
        local group_index = all_attrs[attr_value]

        if not group_index then
            all_attrs.num = 1 + all_attrs.num
            group_index = all_attrs.num
            all_attrs[attr_value] = group_index 
            group_by_attr[group_index] =  {value = attr_value, data = {}}
        end
        table.insert(group_by_attr[group_index].data, x)
    end

    -- compute second term in information gain 
    local weighted_sum = 0 
    for _,group in ipairs(group_by_attr) do
        weighted_sum = weighted_sum + entropy(group.data) * #group.data
    end
    weighted_sum = weighted_sum / #S 
    local gain = entropy_S - weighted_sum
   
    return gain, group_by_attr 
end

local used_attrs = {}

local feature_bagging = params.bagging
local function find_best_split(S)
    -- check all attributes
    -- find ununsed attribute with highest information gain

    if feature_bagging then
        -- randomly select next attr
        local subset_size = torch.floor(0.6 * (#header-1))
        local subset = torch.randperm(#header-1)[{{1,subset_size}}]

        local best_attr, best_grouping
        local highest_gain = -1 
        local all_attrs_used = true 
        for feature_index=1,subset:size(1) do
            local test_attr_index = subset[{feature_index}]
            if not used_attrs[test_attr_index] then
                all_attrs_used = false
                local igain, grouping = information_gain(S, test_attr_index)
                if igain > highest_gain then
                    best_attr = test_attr_index
                    highest_gain = igain
                    best_grouping = grouping
                end
            end
        end
        return best_attr, best_grouping, all_attrs_used
    end

    local best_attr, best_grouping
    local highest_gain = -1 
    local all_attrs_used = true 
    for test_attr_index = 1,#header - 1 do
        if not used_attrs[test_attr_index] then
            all_attrs_used = false
            local igain, grouping = information_gain(S, test_attr_index)
            if igain > highest_gain then
                best_attr = test_attr_index
                highest_gain = igain
                best_grouping = grouping
            end
        end
    end
    return best_attr, best_grouping, all_attrs_used
end

local function majority_vote(data)
    local classes, classes_count = count_classes(data)
    local max_count, max_class = 0
    for _,cl in ipairs(classes) do
        if classes_count[cl] > max_count then
            max_count = classes_count[cl]
            max_class = cl
        end
    end
    --print('majority vote!',classes_count)  
    return max_class
end

local count_nodes = 0 
local function build_tree(node, data, levels)
    count_nodes = count_nodes + 1
    if entropy(data) == 0 then
        node.isleaf = true
        node.class = data[1][#data[1]]
        return 
    end
    local attr_index, grouping, all_attrs_used = find_best_split(data)
    if all_attrs_used or levels == 1 then
        node.isleaf = true
        node.class = majority_vote(data)
        return 
    end
    used_attrs[attr_index] = true
    node.attr_index = attr_index
    node.split_on_attr = header[attr_index]
    node.children = {}
    for i,group in ipairs(grouping) do
        local child = {}
        child.value = group.value
        build_tree(child, group.data, levels-1)
        node.children[i] = child
    end
end

local function evaluate_tree(tree, x)
    if tree.isleaf then
        return tree.class
    end
    local cur_attr = tree.attr_index
    --print('cur attr:', cur_attr, header[cur_attr])
    for _,child in ipairs(tree.children) do
        if child.value == x[cur_attr] then
            return evaluate_tree(child, x)
        end
    end 
    -- return baseline_value
end

-- Build tree(s)
local max_levels = params.levels

num_trees = params.trees 
trees = {}
for t=1,num_trees do
    used_attrs, count_nodes = {}, 0
    local root = {}
    build_tree(root, data, max_levels)
    table.insert(trees, root)
    print('Tree:',t,'Nodes:', count_nodes)
end

-- Test tree
local function test(trees, data)
    local correct = 0
    for i=1,#data do
        local x = data[i]
        local predictions = {}
        for t,tree in ipairs(trees) do
            local prediction = evaluate_tree(tree, x)
            if prediction ~= nil then -- it is possible (and rare) that some samples in test data have unseen
                -- attributes. in such cases, we count on other trees having seen them.
                table.insert(predictions, {prediction})
            end
        end
        local vote = majority_vote(predictions)
        if vote == x[#x] then correct = correct + 1 end
    end
    return correct/#data
end

-- Compute accuracy
local train_acc = test(trees, data)
local test_acc = test(trees, test_data)

print(trees[#trees])

print('Train Accuracy:', 100 * train_acc, '%')
print('Test Accuracy:', 100 * test_acc, '%')

