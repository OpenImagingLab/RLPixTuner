### 
- download yolov3 model yolov3-spp.pt to yolov3/pretrained_models/
## some future direction
- state diff
- agent & training script
- we may add some MoE
- test
  - agent predict all para + stochastic gradient + xx filter
  - agent predict single para + stochastic training + xx filter
  - agent predict single para + deterministic training + xx filter
  - agent MoE
  - agent larger model
  - different state recording
- all disentangle


## update 12 7
- debug : in testing, pick reference, and output param each step to see
  - or actually check evaluation -- redo the evaluation code 
  - done
- accelerate training by moving mem to gpu
  - measure first
'''
your_large_variable = ...  # Your large data structure
memory_size = sys.getsizeof(your_large_variable)
memory_size_kb = memory_size / 1024.0
memory_size_mb = memory_size_kb / 1024.0
'''
- and try to bring in new RL now
- or try your new diffusion + goal guidance way of image manipulation

- actually try output delta first