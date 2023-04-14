# 003--NLP大模型实践--InstructGPT, RLHF and SFT

**参考信息源：**

1. Blog--[Interesting content] [InstructGPT, RLHF and SFT](https://laszlo.substack.com/p/interesting-content-instructgpt-rlhf)
2. Paper--[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)



**Key takeaways:**

- The concept of alignment (a term popularised by Stuart Russell from Berkeley, I link an interview with him in the comments).

  对齐的概念（由伯克利的 Stuart Russell 推广的一个术语，我在评论中链接了对他的采访）。

- InstructGPT is based on GPT-3, but it is aware that it is getting instructions while the older model was only "tricked" into performing them.

  InstructGPT 基于 GPT-3，但它知道它正在获取指令，而旧模型只是被“欺骗”执行指令。

- What is a Reward Model, and how is it incorporated into the training process

  什么是奖励模型，它如何融入培训过程

- The difference between RLHF (reinforcement learning from human feedback) and SFT (Supervised fine-tuning): RLHF is for fine-grain tuning, while SFT causes more significant shift in model behaviour.

  RLHF（从人类反馈中强化学习）和 SFT（监督微调）的区别：RLHF 用于细粒度调整，而 SFT 导致模型行为发生更显着的变化。

- Regarding prompts, the major improvement is that older models needed to be prompted in a specific "almost coded" language, and now they can be prompted more intuitively. They are less sensitive to the prompts but still steerable and, most importantly, "naturally" steerable.

  在提示方面，主要的改进是旧机型需要使用特定的“几乎编码”的语言进行提示，现在可以更直观地进行提示。它们对提示不太敏感，但仍可操纵，最重要的是，“自然地”可操纵。



