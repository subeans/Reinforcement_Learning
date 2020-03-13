'''
게임환경 env의 step() 함수는 우리가 필요로 하는 것을 반환하는데 ' 관찰, 보상, done , info ' 를 반환하게 된다.
1. 관찰 ( observation ) _object : 환경에 대한 관찰을 나타내는 객체이며 환경에 따라 달라진다. 
2. 보상 ( reward ) _float : 이전의 행동을 통해 얻어지는 보상의 양
3.done _boolean : 환경을 reset 해야할지 나타내는 진리값
4. info _ dict : 디버깅할 때 필요한 정보
'''


import gym

env=gym.make('CartPole-v0')
observation = env.reset()
print(observation)  # 카트의 위치, 카트의 속도, 막대기의 각도, 막대기의 회전률 
action = env.action_space.sample()
print(action)   # 0 또는 1의 값이 나옴 
step = env.step(action) 
print("First observation : ",observation )
print("Action :", action )
print("Step :",step)    #step은 action을 선택했을 때 observation,reward,done,info를 반환
                        # done 이 true 인 경우는 terminal ! , info의 정보는 디버깅할 때 사용 많이한다 

for i in range(20):
  observation= env.reset()
  
  for t in range(100):
    #env.render() 영상으로 보여주는 코드인데 서버에서 돌릴경우는 display가 없기때문에 에러가 난다
    print(observation)
    action=env.action_space.sample()
    observation,reward,done,info = env.step(action)
    
    if done:
      print("Episode finished after {} timesteps".format(t+1))
      break
env.close()



'''
env.step(0) 을 하게 되면, 카트의 속도가 왼쪽 방향으로 증가하고 막대기의 회전율이 오른쪽으로 기우는 것을 확인할 수 있다. 
따라서 행동 0 은 왼쪽방향으로 힘을 가하는 것을 알 수 있으므로 행동 1은 오른쪽 방향으로 힘을 가하느 것이다 .
'''

#Cartpole Algorithm

for i in range(100):
  # 막대기가 오른쪽으로 기울어져 있다면, 오른쪽으로 힘을 가하고 그렇지 않다면 왼쪽으로 힘을 가하기
  if observation[2] > 0 : 
    action=1
  else:
    action=0
    
    
  observation, reward, done, info = env.step(action)
  print(observation,done)
  if done:
    print(i+1)  #어느 정도 쓰러지지않고 유지했는가 
    break
env.close()
