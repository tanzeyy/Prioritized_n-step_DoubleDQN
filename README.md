# Prioritized n-step Double DQN

PyTorch implementation of n-step double DQN with prioritized replay memory.



### Experiments

- MoutainCar-V0
  - orange, light blue: step = 1
  - red: step = 5
  - blue: step = 10

<img src="./imgs/image-20201116135106086.png" align="middle">
<img src="./imgs/image-20201116135116360.png" align="middle">
<img src="./imgs/image-20201116135124075.png" align="middle">


- Acrobot-v1
  - n-step from up to bottom: 10, 5, 3, 1

<img src="./imgs/image-20201116135332582.png" align="middle">
<img src="./imgs/image-20201116135344431.png" align="middle">
<img src="./imgs/image-20201116135354706.png" align="middle">


### Requirements
- PyTorch==1.7.0
