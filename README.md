# MATD3
Use Multi-agent Twin Delayed Deep Deterministic Policy Gradient to find collision free paths for ships  
This code is modified based on the privious MADDPG algorithm:https://github.com/Emmanuel-Naive/MADDPG

## Known dependencies: 
  Python: 3.9  
  CUDA: 11.2  
  
  (Python package:)  
  pytorch: 1.10.2  
  tensorboard: 2.9.0  
  numpy: 1.21.5  
  matplotlib: 3.4.1  
  ffmpeg: 2.7.0  
  os, math, random: Python built-in package

## Known issues:
  1. These codes sometimes could only work on 1 core of CPU:  
      Here are some tests on different computer:  
      Computer 1(CPU: Inter i5-6300HQ; GPU: NVIDIA GTX965m): work on all cores  
      Computer 2(CPU: Inter Xeon 5218): work on most cores  
      Computer 3(CPU: Inter Xeon 5218R; GPU: NVIDIA Quadro P2200): **only** work on 1 core  
      Computer 4(CPU: AMD EPYC 7543): **only** work on 1 core  
