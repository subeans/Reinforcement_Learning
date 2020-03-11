'''
Reinforcement Learning DQN Tutorial

OpenAI Gym에서 제공하는 ' cartpole-v0 '을 pytorch를 이용해서 Deep Q-Learning 을 연습한다.
선택하는 action에는 left || right 두가지가 있으며 cart를 움직이고 pole을 서있을 수 있도록 하는 것이 목표이다. 

다양한 알고리즘과 시각화된 것을 Gym website를 통해 공식적인 리더보드를 확인할 수 있다. 

? 리더보드란 ?
: List 나 Search의 프로필, 페이지, 계정 그룹이 특정 기간에 어떤 성과를 올리고 있는지 확인할 수 있다. 경쟁사를 분석할 때 유용하다.
? 언제 사용할까 ?
: 모든 페이지, 프로필 또는 계정의 성과를 한 번에 분석할 때 , 경쟁사의 성과와 내 성과를 비교하여 추적할 때, 브랜디드 콘텐츠의 성과를 추적하고 새로운 교차 게시 파트너를 찾고자 할 때

rewards는 timestep이 증가할 때마다 +1 , 막대가 너무 멀리 떨어지거나 카트가 2.4단위 이상 중앙에서 멀어지면 환경이 종료

Neural networks 에서는 장면을 보고 작업을 해결하며 카트 중심의 화면 패치를 입력으로 사용한다. 
모든 프레임을 렌더링 해야되기 때문에 훈련 속도가 느려진다. 

? 렌더링 ?
: 컴퓨터 프로그램을 사용하여 모델로부터 영상을 만들어내는 과정을 말한다. 
? 패치 ?
: 기능을 더한다, 기본프로그램에 조그만 부분을 덧댄다 , 수정된 부분

현재화면 패치와 이전화면 패치의 차이점으로 상태를 표시한다. 

'''





import gym          # open-AI 에서 만든 gym이란 파이썬 패키지를 이용하여 강화학습 훈련을 할 수 있는 Agent아 Environment를 제공받는다. 
import math
import random
import numpy as np  #python을 통해 데이터 분석을 할 때 기초 라이브러리로 사용되는 numpy, 고성능 수치계산 
import matplotlib
import matplotlib.pyplot as plt     #파이썬에서 데이터를 차트나 plot으로 그려주는 라이브러리 패키지로써 데이터 시각화 패키지
from collections import namedtuple  #namedtuple 은 기본 자료형이 아닌, 튜플의 성질을 가졌지만 항목에 이름으로 접근이 가능한 것을 나타내며 collections는 파이썬 데이터 모델링 파트에서 추천하는 라이브러리
from itertools import count         # python에서 제공하는 자신만의 반복자를 만드는 모듈 itertools , 무한루프를 가정하고 만들어진 라이브러리 
from PIL import Image           #이미지파일 읽고 쓰기 , 모든 이미지를 가지고 하는 작업들 

import torch                   #Pytorch는 텐서플로와 함께 딥러닝 구현에 가장 많이 사용되는 패키지이다. 
import torch.nn as nn           # nn -> neural network
import torch.optim as optim     # optim 다양한 최적화 알고리즘을 정의
import torch.nn.functional as F
import torchvision.transforms as T     #CIFAR10의 학습용, 시험용 데이터셋을 불러오고 정규화


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()   #브라우저에서 그림을 볼 수 있게 바로 그려지도록 해주는 코드
if is_ipython:
    from IPython import display

plt.ion()   #interative-on

# device가 cpu 인지 gpu인지 확인하는 부분 // 
# 우리는 gpu를 이용하기 때문에 cpu인 경우는 코드 돌리지 말아야함 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
? cuda ?
: NVIDIA 에서 개발한 GPU 개발 툴 , GPU를 이용한 프로그래밍을 가능하게 함
많은 양의 연산을 동시에 처리하는 것이 가능

직렬연산 - ex) 재귀연산 , using CPU
병렬연산 - 효과적 , using GPU 

'''

###################################################################################
#Replay Memory
#입력 데이터간의 correlation을 줄이기 위해 사용되는 방법이다. Agent의 경험을 data set에 저장해 두고 data set으로부터 uniform random sampling을 통해 minibatch를 구성하여 학습을 진행한다.
#과거 경험에 대해 반복적인 학습 가능 
###################################################################################


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#환경에서 단일 전이를 나타내는 명명된 튜플 (State, action) 의 쌍을 (next_state, reward) 결과에 맵핑하며 , 상태는 화면 차이 이미지 

class ReplayMemory(object):     #매개변수 어디에쓰임,,?

    def __init__(self, capacity):
        self.capacity = capacity    #?
        self.memory = []    #과거 경험 저장하는 공간 
        self.position = 0       #?

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):   #sample 훈련을 위해 무작위 배치 전환을 선택하는 방법 
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# DQN algorithm
#####################################################################

'''
Pre-processing : 주어진 경험중에서 down-sizing 과 gray scale을 하고 GPU환경에 맞게 square로 crop 한 뒤, last 4 frame을 stack으로 쌓는다.
그 다음 timestep 마다 e-greedy 방식으로 action을 취하고 (random) , 이를 통해 reward와 다음 state 를 pre-processing하여 experience를 구성하고 저장한다.
저장된 sample 을 mini-batch로 취하여 사전에 정의된 loss 를 minimize하도록 한다. DQN network는 CNN을 이용하기 때문에 구조가 CNN과 같다.
'''

class DQN(nn.Module):

    def __init__(self, h, w, outputs):  #CNN과 같은 구조 # height , width
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)  #Conv2d( input 수, output수, 5x5 , 이동보폭 2 ) 
        self.bn1 = nn.BatchNorm2d(16)   #? batchNorm 과정 왜?
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Linear 입력의 연결 숫자는 conv2d 계층의 출력과 입력 이미지의 크기에
        # 따라 결정되기 때문에 따로 계산을 해야한다.
        def conv2d_size_out(size, kernel_size = 5, stride = 2): # ?
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


######################################################################
# Input extraction 입력추출
#######################################################################

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # gym이 요청한 화면은 400x600x3 이지만, 가끔 800x1200x3 처럼 큰 경우가 있습니다.
    # 이것을 Torch order (CHW)로 변환한다.
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # 카트는 아래쪽에 있으므로 화면의 상단과 하단을 제거
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
     # 카트를 중심으로 정사각형 이미지가 되도록 가장자리를 제거
    screen = screen[:, :, slice_range]
    # float 으로 변환하고,  rescale 하고, torch tensor 로 변환
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 크기를 수정하고 배치 차원(BCHW)을 추가
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()


######################################################################
# Training
######################################################################

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9 # 임의의 action을 선택할 확률시작 
EPS_END = 0.05  # EPS_END를 향해 지수적으로 감소 
EPS_DECAY = 200 # 감소 속도 제어 
TARGET_UPDATE = 10

# AI gym에서 반환된 형태를 기반으로 계층을 초기화 하도록 화면의 크기를
# 가져옵니다. 이 시점에 일반적으로 3x40x90 에 가깝다.
# 이 크기는 get_screen()에서 고정, 축소된 렌더 버퍼의 결과이다.
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)     #네트워크 분리 Target값의 불안정성을 해결한 방법 
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000) #10000 무엇?


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택 가능
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


######################################################################
# Training loop
# 최종적으로 모델 학습을 위한 코드 
######################################################################


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산된다.
    # max(1)[0]으로 최고의 보상을 선택
    # 최종인 경우 0 값을 갖는다 
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


#traing loop



num_episodes = 50
for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # 행동 선택과 수행
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 새로운 상태 관찰
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

       # 다음 상태로 이동
        state = next_state

       # 최적화 한단계 수행(목표 네트워크에서)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    #목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
