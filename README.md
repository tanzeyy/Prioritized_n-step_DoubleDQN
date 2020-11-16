# Prioritized n-step Double DQN

PyTorch implementation of n-step double DQN with prioritized replay memory.



### Experiments

- MoutainCar-V0
  - orange, light blue: step = 1
  - red: step = 5
  - blue: step = 10

![image-20201116135106086](./img/image-20201116135106086.png)

![image-20201116135116360](./img/image-20201116135116360.png)

![image-20201116135124075](./img/image-20201116135124075.png)

- Acrobot-v1
  - n-step from up to bottom: 10, 5, 3, 1

![image-20201116135332582](./img/image-20201116135332582.png)

![image-20201116135344431](./img/image-20201116135344431.png)

![image-20201116135354706](./img/image-20201116135354706.png)


### Requirements
- PyTorch==1.7.0