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
  math, os, random, time: Python built-in package
## Scenarios illustration
  The illustration of scenarios could be found in:https://github.com/Emmanuel-Naive/MATD3/blob/main/Scenarios/Scenario_illustration.ipynb  
  **Notice:** the initial data is given based on the XoY coordinate system
## Some results:
  Videos could be found in:https://github.com/Emmanuel-Naive/MATD3/issues/2  
  Or click the url link for wataching.  
  1ShipM: https://user-images.githubusercontent.com/55374976/178238690-7bc441cd-246f-48e2-93e8-bf0a57f53bb6.mp4  
  2Ships_C2: https://user-images.githubusercontent.com/55374976/178238745-945a4c8c-0387-4c6e-8705-582052eaa9f0.mp4  
  2Ships_H2: https://user-images.githubusercontent.com/55374976/178238743-52e98bc4-4d64-4d7c-9234-011839d61cad.mp4  
  2Ships_O2_Right: https://user-images.githubusercontent.com/55374976/178238736-4d851554-a85d-44b0-a0a7-1bba7a1c21dd.mp4  
  2Ships_O2_Left: https://user-images.githubusercontent.com/55374976/178238740-ddfb4cca-5592-49a3-aa3d-3aa9f527ff4b.mp4  
  3Ships_C3H2: https://user-images.githubusercontent.com/55374976/178238783-67965cbe-9130-4404-9cc3-6175900fc739.mp4   
  3Ships_C3O2: https://user-images.githubusercontent.com/55374976/178238786-36ede5d5-90d5-47c2-b9d3-572eaaf90839.mp4  
  3Ships_H3O2: https://user-images.githubusercontent.com/55374976/178238792-bacca074-14c8-49f0-bbb8-e6d698bac7cc.mp4  
  4Ships_C4H3O2: https://user-images.githubusercontent.com/55374976/178238794-a3b923b2-ddde-4284-8b03-f04c099771d8.mp4  
  4Ships_C4H4: https://user-images.githubusercontent.com/55374976/178238797-4d39941c-0ffa-4710-b2c0-015b96cb8b29.mp4  

## Known issues:
  1. These codes sometimes could only work on 1 core of CPU, but I am not sure that parallel processing could be used in deep reinforcement learning because data in each episode can not be processed individually. However, data for each vessel may be able to be processed individually.
      Here are some tests on different computer:  
      Computer 1(CPU: Inter i5-6300HQ; GPU: NVIDIA GTX965m): work on all cores  
      Computer 2(CPU: Inter Xeon 5218): work on most cores  
      Computer 3(CPU: Inter Xeon 5218R; GPU: NVIDIA Quadro P2200): **only** work on 1 core  
      Computer 4(CPU: AMD EPYC 7543): **only** work on 1 core  
        
     However, if the code must be run **step by step, episode by episode**, this issues does not matter because codes could not be parallel computed.  
     Another thought is that in computer 2, those CPU cores are used for calculation of networks, but in computer 1&3, this work would be done by GPUs.
     Some tests results are here(test on computer 1):  
     ```
     (10000 episodes normally need to take 2 days on calculation)  
     For each episode (need to train 10000 episodes):
        (each episode would take 4-20s. If reaching the maximum steps, this episode would take around 20s)  
        For each step (jump to next episode if reaching the maximum limit or finishing the goal): 
            (each step would take 0.016-0.032s. If the learn_function use backward() function twice, this step would take 0.0312s)  
             action_function (would take 0.002s)  
             step_function (would take 0.001s)  
             learn_function (would take 0.0156s or 0.0312s, after the GPU acceleration)  
             reward_function (would take 0.000s)  
             norm_function (would take 0.000s)  
             store_function (would take 0.000s)
  2. The error happens when using backward() for optimizing actors.  
  ```RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.```  
 Details:   
 https://discuss.pytorch.org/t/i-am-training-my-multi-agents-reinforcement-learning-project-and-i-got-an-error-trying-to-backward-through-the-graph-a-second-time/152352  
 https://discuss.pytorch.org/t/when-and-why-do-i-need-to-use-detach-in-loss-calculating-and-backpropagation/152683
