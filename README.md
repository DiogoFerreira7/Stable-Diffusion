# Stable Diffusion Pipeline

The goal of this project was to get a thorough and deep understanding of the mathematics but also the implementation of the original Stable Diffusion as well as other newer advances in the field like DDIM. 

With this goal in mind I have created several visualisation methods that are all explained below that breakdown every concept at different steps in the pipeline.
You can visualise the initial tokenisation and attention mask where CLIP guides the diffusion process, visualise the latents of any generated image, plot how different schedulers control sigma (and hence the noise) as well as view the noised images affected by any noise schedule.

#### Project Files: 
- m.py - scrip
- gt.py - tra

Skip [here](#results) to see the of my GPT 3 Large (0.76B param) model

## How to

### Visualise latents

Loading 

![GPT-2 Pretrained](assets/pretrained_gpt.png)

### Visualise tokenisation and attention masks

To train your

![GPT2 Configuration](assets/gpt2_config.png)

Tips:
- You
- Che

## Resources

### Useful Papers I Read

- [Attention is all you need](https://arxiv.org/pdf/1706.03762)

- [Flash attention](https://arxiv.org/pdf/2205.14135)

- [Flash attention 2](https://arxiv.org/pdf/2307.08691)

- [Language models are unsupersived multitask learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- [Language models are few-shot learners](https://arxiv.org/pdf/2005.14165)

- [GPT-4 technical report](https://arxiv.org/pdf/2303.08774)

### Other sources

These resources were of great help in understanding the inner workings behind transformers and the GPT papers.

3Blue1Brown - [Mathematics behind transformers - Chapter 5/6](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=7kUJ3D5-B24sOq7j)

OpenAI - [GPT-2 Tensorflow Implementation](https://github.com/openai/gpt-2/blob/master/src/model.py)

Hugging Face Tranformers - [GPT-2 PyTorch Implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

PyTorch - [Fast GPT Implementation and Article](https://github.com/pytorch-labs/gpt-fast)

Andrej Karpathy - [Zero to Hero: NN Course](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=xMIrxu1JbABFPRej)

## Common Problems & Fixes

Making sure 
- [Comprehensive StackOverflow Guide](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)

If you are
- To solve this either set


