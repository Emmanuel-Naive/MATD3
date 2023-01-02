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
  The illustration of scenarios could be found in:https://github.com/Emmanuel-Naive/Illustration-of-scenarios-MATD3-/blob/main/Scenario_illustration.ipynb  
  **Notice:** the initial data is given based on the XoY coordinate system  
  **PS:** Explanation for scenario names  
  &emsp;xShips: The total number of ships in the given scenario
  &emsp;C:Crossing scenario  
  &emsp;H:Head-on scenario  
  &emsp;O:Overtaking scenario  
  &emsp;Number after letters(C,H,O): The number of ships in the given scenario represented by the letter 
  &emsp;Example: Scenario 3Ships_H3O2: (3Ships) In this scenario, there are 3 ships. (H3) All 3 ships are set in head-on scenarios while (O2) 2 ships are set in the overtaking scenario.  
## Some results:
  Videos could be found in:https://github.com/Emmanuel-Naive/MATD3/issues/2  
  Or click the url link for wataching.  
  
  1ShipM: https://user-images.githubusercontent.com/55374976/178543309-ceeac31a-345e-4de0-ad66-9caebc4bb0ac.mp4  
  2Ships_H2: https://user-images.githubusercontent.com/55374976/178543347-816bbda8-b8ef-45b0-9246-de11124b6761.mp4  
  2Ships_C2: https://user-images.githubusercontent.com/55374976/178543349-22b1fc0d-52f3-48eb-ba9d-79673cdb8a23.mp4  
  2Ships_O2_Right: https://user-images.githubusercontent.com/55374976/178543339-38dc1018-7c63-48f2-b6ac-a1ec78be7031.mp4  
  2Ships_O2_Left: https://user-images.githubusercontent.com/55374976/178543344-26571480-8463-411d-b0d2-df32215db90a.mp4  
  3Ships_C3H2: https://user-images.githubusercontent.com/55374976/178543369-7389dd68-a0b3-4d7c-85de-9cccf783e978.mp4  
  3Ships_C3O2: https://user-images.githubusercontent.com/55374976/178543375-ce3eb21e-69fb-43f0-be2f-173905220dfb.mp4  
  3Ships_H3O2: https://user-images.githubusercontent.com/55374976/178543377-2553abe6-446a-4931-9da2-96d7bb491f4a.mp4  
  4Ships_C4H3O2: https://user-images.githubusercontent.com/55374976/178543378-17cee073-b5f6-4718-83d5-6448e30eb7d2.mp4  
  4Ships_C4H4: https://user-images.githubusercontent.com/55374976/178543381-19224e71-d880-4943-8af5-6247beca6e57.mp4  
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
