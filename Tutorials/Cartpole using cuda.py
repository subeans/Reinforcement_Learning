import gym
from gym import wrappers
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# hyper parameters
EPISODES = 500  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.99  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 32  # Q-learning batch size

#batch_size , gamma , learning rate 가 RL에 영향 

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
        
       
env = gym.make('CartPole-v1')
#env = wrappers.Monitor(env, './tmp/cartpole-v0-1')

model = Network()
if use_cuda:
    model.cuda()
memory = ReplayMemory(1000)
#ReplayMemory의 capacity 의 크기에 따라 RL이 잘이루어지는데 영향 
optimizer = optim.Adam(model.parameters(), LR)
#torch.optim.Adam( PARAMS , LR = 0.001 , 베타 = (0.9 , 0.999) , EPS = 1E-08 , weight_decay = 0 , amsgrad = 거짓 )
# params : 매개변수 그룹 정의를 최적화하거나 지시하기위한 매개변수의 반복 기능, lr : 학습 속도 , 베타 : 그라디언트의 실행 평균 및 제곱의 계산에 사용되는 계수
# eps : 숫자 안정성을 향상시키기위해 분모에 추가된 항 
steps_done = 0
episode_durations = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:  # e보다 크면 model 에서 큰 값을 따르고
        return model(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        # Variable () : 모델을 훈련할 때 파라미터를 저장할 변수가 필요하고 메모리에 저장할 수 있도록해주는 것
        # gradient 가 필요한 input 인 경우, output도 gradient가 필요하므로 backward연산을 해야하고 
        # input output 둘 다 gradient가 필요없는 경우는 backward 연산은 이루어지지 않는다. 
    else:       # e보다 작으면 random한 값을 사용한다 
        return LongTensor([[random.randrange(2)]])


def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    while True:
        #environment.render()
        action = select_action(FloatTensor([state]))
        next_state, reward, done, _ = environment.step(action.item())

        # negative reward when attempt ends
        if done:
            reward = -1

        memory.push((FloatTensor([state]),
                     action,  # action is already a tensor
                     FloatTensor([next_state]),
                     FloatTensor([reward])))

        learn()

        state = next_state
        steps += 1

        if done:
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            episode_durations.append(steps)
            plot_durations()
            break


def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
    
    #복수의 tensor을 결합할때 torch.cat
    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
       
for e in range(EPISODES):
    run_episode(e, env)

print('Complete')
#env.render(close=True)
env.close()
plt.ioff()
plt.show()
