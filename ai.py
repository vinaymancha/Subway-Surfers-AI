
# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import time



import experience_replay

def save():
        torch.save({'state_dict': cnn.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                   }, 'last_brain.pth')
    
def load():
    if os.path.isfile('last_brain.pth'):
        print("=> loading checkpoint... ")
        checkpoint = torch.load('last_brain.pth')
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("done !")
    else:
        print("no checkpoint found...")





# Initializing the weights of the neural network in an optimal way for the learning
        
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        
        
# AI

# Making the brain

class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.out_neurons = self.count_neurons((1, 128, 128))
        self.lstm = nn.LSTMCell(self.out_neurons,256)
        #self.fc1 = nn.Linear(in_features = self.count_neurons((1, 256, 256)), out_features = 62)
        self.fc2 = nn.Linear(in_features = 256, out_features = number_actions)
        self.apply(weights_init)
        self.fc2.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.elu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.elu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.elu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x, hidden = None):
        x = x.cuda()
        if isinstance(hidden, tuple):
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(-1, self.out_neurons)
        hx, cx = self.lstm(x, hidden)
        x = hx
        #x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, (hx, cx)

# Making the body

class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=0)   
        actions = probs.multinomial(num_samples=1)
        return actions

# Making the AI

class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs, hidden):
        output, (hx, cx) = self.brain(inputs, hidden)
        actions = self.body(output)
        return actions.data.cpu().numpy(), (hx, cx)



# Training the AI with Deep Convolutional Q-Learning


# Building an AI
cnn = CNN(number_actions=5)
cnn = cnn.to("cuda:0")
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress( ai = ai, n_step = 5)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 1000)
    
# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output, hidden = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)


# Making the moving average on 100 steps
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)
ma = MA(100)            

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.0007)
nb_epochs = 70
#load()           ##To load previous weights
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(100)
    print("Entering epoch")
    for batch in memory.sample_batch(72):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions, hidden = cnn(inputs, None)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    save()
    time.sleep(4)
    if avg_reward >= 200:
        print("Congratulations, your AI wins")
        save()                
        break


