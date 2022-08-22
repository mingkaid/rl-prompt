# RL Prompt

This repo contains the code of the discrete prompt optimization framework described in the paper \
**[RLPrompt: Optimizing Discrete Text Prompts With Reinforcement Learning](https://arxiv.org/abs/2205.12548)** \
Mingkai Deng*,   Jianyu Wang*,   Cheng-Ping Hsieh* (equal contribution),   Yihan Wang,   Han Guo,   Tianmin Shu,   Meng Song,   Eric P. Xing,   Zhiting Hu 

We will keep updating the codebase for easier usage and adaptation for your own tasks, so please stay tuned by starring or watching our repo! 

## Getting Started

* Extensive recent work (e.g., [this](https://arxiv.org/abs/2107.13586)) has shown that *prompting* pre-trained LMs with specific text can steer them to perform various NLP tasks, without needing to update the model
* Previous work has typically tuned soft prompts with gradient-based optimization or searched for discrete text prompts using various heuristics
* In our paper, we propose to formulate discrete prompt optimization as an RL problem, and train a policy network to generate the prompt that optimizes a reward function
* Compared to typical soft prompts, our discrete prompts are lightweight, interpretable, and transferrable across model types (e.g., RoBERTa to GPT-2) and sizes (e.g., small to large)
* Check out more analyses at our paper [here](https://arxiv.org/abs/2205.12548)

![](figure.png)

## Setting Up

Our codebase requires the following Python and PyTorch versions: 
* Python >= 3.7
* PyTorch >= 1.10.1 (install from the [official website](https://pytorch.org/get-started/locally/))

Install our core modules with
```
pip install -e .
```

## Usage

Please refer to the folders in `examples`, which contains our implementations of 1) [few-shot classification](https://github.com/mingkaid/rl-prompt/tree/main/examples/few-shot-classification) and 2) [text style transfer](https://github.com/mingkaid/rl-prompt/tree/main/examples/text-style-transfer), as described in our paper.  

In short, the code in `rlprompt` provides the core components for prompt optimization. The task-specific folders in `examples` simply implement the reward functions and use the core modules to run experiments.  
