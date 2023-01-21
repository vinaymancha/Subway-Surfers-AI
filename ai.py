
# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import os
import time



import experience_replay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hx = torch.empty(1, 256).to(device)
cx = torch.empty(1, 256).to(device)

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
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
        
        
# AI

# Making the brain



class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        self.mobilenet = self.mobilenet.to(device)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc2 = nn.Linear(in_features = 1000, out_features = number_actions)
        self.apply(weights_init)
        self.fc2.bias.data.fill_(0)
        self.train()

    def forward(self, x, hidden = None):
        x = x.to(device)
        x = self.mobilenet(x)
        x = self.fc2(x)
        return x, (hx,cx)

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
cnn = cnn.to(device)
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
    memory.n_steps_iter = iter(memory.n_steps)
    memory.run_steps(100)
    print("Entering epoch")
    for batch in memory.sample_batch(36):
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



