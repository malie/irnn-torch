-- Re-implementation by Markus Liedl of "The Sum Experiment" from
-- "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
-- by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton
-- https://arxiv.org/pdf/1504.00941.pdf

io.stdout:setvbuf("no")
require 'torch'
require 'nn'
require 'nnx'
require 'optim'
local inspect = require 'inspect'

opt = {
   learningRate = 1e-4,
   numHidden = 100,
   numTimesteps = 80, -- 240, -- 160,
   batchsize = 64,
   gpu = true,
   gc = 1.0, -- gradient clipping
   cheatFinalBias = true
}

print(inspect(opt))

if opt.gpu then
   -- not tested without gpu...
   require 'cudnn'
   require 'cunn'
end

numInputs = 2

function initIdentity(lin)
   local w = lin.weight

   -- init recurrent weights a diagonal matrix
   local recurrentWeights = w[{{}, {1, opt.numHidden}}]
   torch.eye(recurrentWeights, opt.numHidden)

   -- recurrentWeights:mul(0.99)

   -- init input weights to normal distribution
   w[{{}, {opt.numHidden+1, opt.numHidden+numInputs}}]:normal(0, 0.001)
   return lin
end

function fetchInput(input, output)
   -- print(input:size())
   assert(input:size(1) == opt.numTimesteps)

   -- sample indices for both flagged inputs; make sure they are different!
   local ra = torch.random(1, opt.numTimesteps)
   local rb
   while true do
      rb = torch.random(1, opt.numTimesteps)
      if rb ~= ra then
         break
      end
   end

   local sum = 0.0
   for t = 1, opt.numTimesteps do
      local xin = t == ra or t == rb
      local val = torch.uniform()
      local trig = 0
      if xin then
         trig = 1.0
         sum = sum + val
      end
      input[{t, 1}] = val
      input[{t, 2}] = trig
   end
   output[1] = sum
end


-- calc new hiddens; applied at each timestep
hiddenNet = nn.Sequential()
hiddenNet:add(initIdentity(nn.Linear(opt.numHidden + numInputs, opt.numHidden)))
hiddenNet:add(nn.ReLU())

print('hiddenNet')
print(hiddenNet)


-- calc final output; applied after all the timesteps
final = nn.Sequential()
local lin1 = nn.Linear(opt.numHidden, 200)
final:add(lin1)
lin1.weight:normal(0, 1./math.sqrt(200))
final:add(nn.ReLU())

-- final:add(nn.BatchNormalization(200))

local lin = nn.Linear(200, 1)
-- lin.weight:normal(0, 0.04)

-- let's help the final layer by setting the bias to the mean of the sum of two
-- uniform random numbers
if opt.cheatFinalBias then
   lin.bias[1] = 1.0
end

final:add(lin)

-- -- might use a single layer final layer:
-- local lin = nn.Linear(opt.numHidden, 1)
-- lin.weight:normal(0, 0.01)
-- lin.bias[1] = 1.0
-- final:add(lin)

print('final')
print(final)

criterion = nn.MSECriterion()

if opt.gpu then
   hiddenNet:cuda()
   final:cuda()
   criterion:cuda()
end


function concatHiddenAndInput(hiddensAndInput, hidden, input)
   hiddensAndInput[{ {}, {1, opt.numHidden}}]:copy(hidden)
   hiddensAndInput[{ {}, {opt.numHidden+1, opt.numHidden+numInputs}}]:copy(input)
end


local input = torch.Tensor(opt.batchsize, opt.numTimesteps, 2)
local target = torch.Tensor(opt.batchsize, 1):cuda()
local hin = torch.Tensor(opt.batchsize, opt.numHidden + numInputs)
local allInputs = torch.Tensor(opt.numTimesteps, opt.batchsize, opt.numHidden + numInputs)
local hidden = torch.Tensor(opt.batchsize, opt.numHidden):zero()

if opt.gpu then
   input = input:cuda()
   target = target:cuda()
   allInputs = allInputs:cuda()
   hin = hin:cuda()
   hidden = hidden:cuda()
end

-- `all' is only used for getParameters()
all = nn.Sequential()
all:add(hiddenNet)
all:add(final)
parameters, gradParameters = all:getParameters()

sgdState = {
  learningRate = opt.learningRate / 100,
  -- beta1 = 0.5,
}

mb = 1
avgErr = 0.0
while true do
   local verbose = mb % 100 == 1

   local feval = function (x)
      for i = 1, opt.batchsize do
         fetchInput(input[i], target[i])
      end

      hidden:zero()
      
      for t = 1, opt.numTimesteps do
         local tInput = input[{ {}, t, {}}]
         concatHiddenAndInput(hin, hidden, tInput)

         -- save input for backward pass later
         allInputs[{ t, {}, {}}]:copy(hin)
   
         local hout = hiddenNet:forward(hin)
         if verbose then
            -- print deviation of activations
            print('--------------- ' .. t .. ' ' .. torch.std(hout))
         end
         hidden = hout
      end

      local output = final:forward(hidden)
      -- print('final output', output)
      -- print('target', target)

      local err = criterion:forward(output, target)
      local grad = criterion:backward(output, target)

      local f = 0.99
      if mb < 100 then f = 0.9 end
      avgErr = avgErr * f + err * (1-f)
      print(string.format('%5.3f mb %6i err %1.7f avg err %1.7f', os.clock(), mb, err, avgErr))
      if verbose then
         print('grad', torch.std(grad))
      end

      gradParameters:zero() -- reset gradients

      final:backward(hidden, grad)
      grad = final.gradInput

      for t = opt.numTimesteps, 1, -1 do
         local input = allInputs[{t, {}, {}}]

         hiddenNet:forward(input) -- TODO: is this necessary???
         local hout = hiddenNet:backward(input, grad)

         -- get previous timestep grad with respect to the hiddens there
         -- (drop the gradient with respect to the inputs, the last two)
         grad = hiddenNet.gradInput
         grad = grad[{{}, {1, opt.numHidden}}]

         if verbose then
            -- print deviation of gradient
            print('--------------- grad ' .. t .. ' ' .. torch.std(grad))
         end
      end

      gradParameters:clamp(-opt.gc, opt.gc)

      return nil, gradParameters
   end

   -- start using full learning rate after some steps
   if mb == 10 then
      sgdState.learningRate = opt.learningRate
   end

   optim.rmsprop(feval, parameters, sgdState)
   mb = mb + 1
end

-- convergence (avg err below 1.59) starts
--   for  80 timesteps at mb 4400,
--   for 120 timesteps at mb 8200
--   for 160 timesteps at mb 11200
--   for  40 timesteps at mb 3200 (with opt.cheatFinalBias off)
--   for  80 timesteps at mb 4800 (with opt.cheatFinalBias off)
--   for 160 timesteps at mb 9901 (with opt.cheatFinalBias off)
