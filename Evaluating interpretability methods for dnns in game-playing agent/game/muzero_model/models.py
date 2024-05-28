import torch
import torch.nn as nn
import torch.nn.functional as F

from muzero_model.breakthrough import State

import numpy as np


class ResidualBlock(nn.Module):
  def __init__(self, channels=32):
    super().__init__()
    self.block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                               nn.BatchNorm2d(channels),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(channels, channels, 3, padding=1),
                               nn.BatchNorm2d(channels))
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x):
    residual = x
    x = self.block(x)
    x += residual
    x = self.activation(x)
    return x

class ConvBlock(nn.Module):
  def __init__(self, in_channels=17, out_channels=32):
    super().__init__()
    self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True))

  def forward(self, x):
    return self.block(x)

class Repr_Net(nn.Module):
  def __init__(self, 
               start_channels=17, 
               res_block_channels=32, 
               num_res_blocks=3):
    super().__init__()
    self.res_blocks = nn.Sequential(*[ResidualBlock(res_block_channels) for _ in range(num_res_blocks)])
    self.conv_block = ConvBlock(start_channels,res_block_channels)
  def forward(self, x):
    out = self.conv_block(x)
    out = self.res_blocks(out)
    return out
  def inference(self, x):
    self.eval()
    with torch.no_grad():
      if next(self.parameters()).device.type == 'cuda':
        rp = self(torch.from_numpy(x).unsqueeze(0).to("cuda:0"))
      else:
        rp = self(torch.from_numpy(x).unsqueeze(0)) # .to("cuda:0")
    return rp.cpu().numpy()[0]

class Dynamics_Net(nn.Module):
  def __init__(self, 
               num_res_blocks=3, 
               action_channels=8, 
               res_block_channels=32, 
               image_size=(8,8), 
               intermediate_rewards=False):
      
    super().__init__()
    assert res_block_channels >= 16
    self.res_blocks = nn.Sequential(*[ResidualBlock(res_block_channels) for _ in range(num_res_blocks)])
    self.conv_block = nn.Sequential(nn.Conv2d(action_channels + res_block_channels, res_block_channels, 1),
                                    nn.BatchNorm2d(res_block_channels),
                                    nn.ReLU(inplace=True))

  def forward(self, rp, a):
    x = torch.cat([rp, a], dim=1)

    #def forward(self, x):
    out = self.conv_block(x)
    state = self.res_blocks(out)

    return state #, reward

  
  def inference(self, rp, a):
    self.eval()
    with torch.no_grad():
      if next(self.parameters()).device.type == 'cuda':
        rp = self(torch.from_numpy(rp).unsqueeze(0).to("cuda:0"), 
                  torch.from_numpy(a).unsqueeze(0).to("cuda:0").float()) 
      else:
        rp = self(torch.from_numpy(rp).unsqueeze(0), 
                  torch.from_numpy(a).unsqueeze(0)) 
    return rp.cpu().numpy()[0]

class Predict_Net(nn.Module):
  def __init__(self, 
               num_res_blocks=5, 
               num_actions=512, 
               res_block_channels=32, 
               image_size=(8,8)):
      
    super().__init__()
    assert res_block_channels >= 16
    # assert image_size*image_size*res_block_channels == num_actions
    self.res_blocks = nn.Sequential(*[ResidualBlock(res_block_channels) for _ in range(num_res_blocks)])
    self.action_head = nn.Sequential(nn.Conv2d(res_block_channels, int(num_actions/(image_size[0]*image_size[1])), 1),
                                    nn.BatchNorm2d(int(num_actions/(image_size[0]*image_size[1]))),
                                    nn.ReLU(inplace=True))
    self.value_head = nn.Sequential(nn.Conv2d(res_block_channels, res_block_channels//16, 1),
                                      nn.BatchNorm2d(res_block_channels//16),
                                      nn.ReLU(inplace=True))
    self.fc1 = nn.Linear((res_block_channels//16) * image_size[0] * image_size[1], 8)
    self.fc2 = nn.Linear(8, 1)
    self.activation = nn.ReLU(inplace=True)
    self.tanh = nn.Tanh()
    self.num_actions=num_actions
    self.softmax = nn.Softmax(dim=1)
    self.image_size = image_size
    self.res_block_channels = res_block_channels

  def forward(self, x):
    out = self.res_blocks(x)
    actions = self.action_head(out)
    actions = actions.view(-1, self.num_actions)

    value = self.value_head(out)
    value = value.view(-1, self.res_block_channels//16 * self.image_size[0] * self.image_size[1])
    value = self.fc1(value)
    self.activation(value)
    value = self.fc2(value)

    return nn.functional.softmax(actions, dim=-1), self.tanh(value)

      

  def inference(self, rp):
    self.eval()
    with torch.no_grad():
      
      if next(self.parameters()).device.type == 'cuda':
        p, v = self(torch.from_numpy(rp).unsqueeze(0).to("cuda:0"))
      else:
        p, v = self(torch.from_numpy(rp).unsqueeze(0)) 
    return p.cpu().numpy()[0], v.cpu().numpy()[0][0]


class Nets(nn.Module):
    '''Whole nets'''
    def __init__(self,
                  num_blocks = 5,
                  num_filters = 16):
        super().__init__()
        """
        To construct the net we need first of all to know:
            Shape of the state representation 
            size of the action space
            
        Hyperparameters are:
            num_blocks - number of residual blocks in each net
            num_filters - number of filters in each residual block
        
        """
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.model_name = 'muzero'
        state = State()
        input_shape = state.feature().shape
        action_shape = state.action_feature(0).shape
        rp_shape = (num_filters, *input_shape[1:])

        self.representation = Repr_Net(start_channels=input_shape[0], 
                                       res_block_channels=num_filters, 
                                       num_res_blocks=num_blocks)
        
        self.prediction = Predict_Net(num_res_blocks=num_blocks, 
                                      num_actions=np.prod(action_shape), 
                                      res_block_channels=num_filters, 
                                      image_size=input_shape[1:])
        
        self.dynamics = Dynamics_Net(num_res_blocks=num_blocks, 
                                     action_channels=action_shape[0], 
                                     res_block_channels=num_filters, 
                                     image_size=input_shape[1:])

    def predict_all(self, state0, path):
        '''Predict p and v from original state and path'''
        outputs = []
        self.eval()
        if next(self.parameters()).device.type == 'cuda':
            x = torch.from_numpy(state0.feature()).unsqueeze(0).float().to("cuda:0")
        else:
            x = torch.from_numpy(state0.feature()).unsqueeze(0).float()
        
        with torch.no_grad():
            rp = self.representation(x)
            outputs.append(self.prediction(rp))
            for action in path:
                a = state0.action_feature(action).unsqueeze(0)
                rp = self.dynamics(rp, a)
                outputs.append(self.prediction(rp))
        #  return as numpy arrays
        return [(p.cpu().numpy()[0], v.cpu().numpy()[0][0]) for p, v in outputs]
    


#%%

class AlphaZero(nn.Module):
  def __init__(self, 
               num_blocks = 10,
               num_filters = 128):
      
    super().__init__()
    
    self.model_name = 'alphazero'
    state = State()
    input_shape = state.feature().shape
    action_shape = state.action_feature(0).shape
    num_actions = np.prod(action_shape)
    image_size=input_shape[1:]
    assert num_filters >= 16    
    self.conv_block = ConvBlock(input_shape[0],num_filters)
    self.res_blocks = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_blocks)])
    self.action_head = nn.Sequential(nn.Conv2d(num_filters, int(num_actions/(image_size[0]*image_size[1])), 1),
                                    nn.BatchNorm2d(int(num_actions/(image_size[0]*image_size[1]))),
                                    nn.ReLU(inplace=True))
    self.value_head = nn.Sequential(nn.Conv2d(num_filters, num_filters//16, 1),
                                      nn.BatchNorm2d(num_filters//16),
                                      nn.ReLU(inplace=True))
    self.fc1 = nn.Linear((num_filters//16) * image_size[0] * image_size[1], 8)
    self.fc2 = nn.Linear(8, 1)
    self.activation = nn.ReLU(inplace=True)
    self.tanh = nn.Tanh()
    self.num_actions=num_actions
    self.softmax = nn.Softmax(dim=1)
    self.image_size = image_size
    self.num_filters = num_filters

  def forward(self, x):
    x = self.conv_block(x)
    out = self.res_blocks(x)
    actions = self.action_head(out)
    actions = actions.view(-1, self.num_actions)

    value = self.value_head(out)
    value = value.view(-1, self.num_filters//16 * self.image_size[0] * self.image_size[1])
    value = self.fc1(value)
    self.activation(value)
    value = self.fc2(value)

    return nn.functional.softmax(actions, dim=-1), self.tanh(value)


  def inference(self, rp):
    self.eval()
    with torch.no_grad():
      
      if next(self.parameters()).device.type == 'cuda':
        p, v = self(torch.from_numpy(rp).unsqueeze(0).to("cuda:0"))
      else:
        p, v = self(torch.from_numpy(rp).unsqueeze(0)) 
    return p.cpu().numpy()[0], v.cpu().numpy()[0][0]


class Alphazero_wrapper(nn.Module):
    def __init__(self,                 
                num_blocks = 10,
                num_filters = 128):
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        super().__init__()
        self.prediction = AlphaZero(num_blocks = num_blocks,
                                    num_filters = num_filters)
        self.model_name = self.prediction.model_name
    
    def representation(self, x):
        return x


#%%
if __name__ == '__main__':
  # Check to make sure our networks are working

  from muzero_nets import Nets as Nets2


  net = Nets()#.to("cuda:0")
  state = State()
  import time 
  t0 = time.time()
  for i in range(1000):
    a = net.predict_all(state, [])

  t1 = time.time()
  print('Elapsed: ', t1-t0)
  #print('Net output: ',a)


  if 0:
      
      #%%
    import time
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device_name = "cpu"
    
    
    hidden_state_channels = 100
    res_blocks = 10
    actions = 8
    
    rn = Repr_Net(start_channels=res_blocks, 
                  res_block_channels=hidden_state_channels, 
                  num_res_blocks=3).to(device_name)
    input_rn = torch.randn((1,res_blocks,8,8), device = device_name)
    
    dn = Dynamics_Net(num_res_blocks=res_blocks, 
                      action_channels=actions, 
                      res_block_channels=hidden_state_channels).to(device_name)
    input_dn = torch.randn((1,hidden_state_channels,8,8)).to(device_name)
    a_dn = torch.randn((1,actions,8,8), device = device_name)
    
    pn = Predict_Net(num_res_blocks=res_blocks, 
                     num_actions=512, 
                     res_block_channels=hidden_state_channels).to(device_name)
    input_pn = torch.randn((1,hidden_state_channels,8,8), device = device_name)
    
    t00 = time.time()
    for i in range(100):
        out = rn(input_rn)
    print("Output from Repr_Net:", out.shape)
    t01 = time.time()
    print(t01-t00)

    for i in range(100):
        state = dn(input_dn, a_dn)
    print("Output from Dynamics_Net:", state.shape)
    t02 = time.time()
    print(t01-t00)

    for i in range(100):
        actions, values = pn(input_pn)
    print("Output from Predict_Net:", actions.shape, values.shape)
    
    
    t03 = time.time()
    print(t03-t02)
    #   print(pn)
    
#%%

    num_filters = 16
    num_blocks = 5
    state = State()
    input_shape = state.feature().shape
    action_shape = state.action_feature(0).shape
    rp_shape = (num_filters, *input_shape[1:])

    representation = Repr_Net(start_channels=input_shape[0], 
                                   res_block_channels=num_filters, 
                                   num_res_blocks=num_blocks)
    
    prediction = Predict_Net(num_res_blocks=num_blocks, 
                                  num_actions=np.prod(action_shape), 
                                  res_block_channels=num_filters, 
                                  image_size=input_shape[1:])
    
    #%%
    f = state.feature()
    rp = representation(torch.from_numpy(f).unsqueeze(0))
    
    
    
    #%%
    x = rp
    out = prediction.res_blocks(x)
    print('out shape: ',out.shape)
    actions = prediction.action_head(out)
    print('1 actions shape: ',actions.shape)
    actions = actions.view(-1, prediction.num_actions)
    print('2 actions shape: ',actions.shape)

    value = prediction.value_head(out)
    print('1 value shape: ',value.shape)
    value = value.view(-1, prediction.res_block_channels//16 * prediction.image_size[0] * prediction.image_size[1])
    print('2 value shape: ',value.shape)
    value = prediction.fc1(value)
    print('3 value shape: ',value.shape)
    prediction.activation(value)
    print('4 value shape: ',value.shape)
    value = prediction.fc2(value)
    print('5 value shape: ',value.shape)
    
    
    #%%
    filter_sizes = [16,32,64,128, 264]
    block_numbers = [1,3,5,10]
    num_runs = 1000
    cols = 5
    rows = 6
    runtimes = []
    for filter_size in filter_sizes:
        for block_num in block_numbers:
            res_blocks = nn.Sequential(*[ResidualBlock(filter_size) for _ in range(block_num)])
            res_blocks.cuda()
            arr = np.random.rand(1,filter_size, rows, cols)
            tensor_ = torch.from_numpy(arr).float().cuda()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)            
            start.record()
            for i in range(num_runs):
                out = res_blocks(tensor_)
            end.record()
            
            # Waits for everything to finish running
            torch.cuda.synchronize()
            
            t = start.elapsed_time(end)
            
            runtimes.append((filter_size, block_num, t))
            
    #%%
    
    import pandas as pd
    
    df = pd.DataFrame(runtimes, columns =  ['filters','blocks','ms'])
    table = pd.pivot_table(df, values='ms', index=['blocks'], columns=['filters'])
    
    
    
    
            
            
    
    
    
    
    
    
    
    
    