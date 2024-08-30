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

- [U-Net](https://arxiv.org/pdf/1505.04597)

- [Sigmoid Linear Unit - SiLU](https://paperswithcode.com/method/silu)

- [Stats behind stable diffusion](https://mbernste.github.io/posts/diffusion_part1/)

- [Original Paper](https://arxiv.org/pdf/1503.03585.pdf)

- [DDPM](https://arxiv.org/pdf/2006.11239.pdf)

The following papers I found give improvements to the models and were both released in 2021
- [Paper 1](https://arxiv.org/pdf/2102.09672.pdf)
- [Paper 2](https://arxiv.org/pdf/2105.05233.pdf)

### Other sources

I would highly recommend reading these to get a good understanding.

- [OpenAI CLIP Explanation](https://openai.com/index/clip/)

- [Mathematics Behind DDPM (YouTube)](https://www.youtube.com/watch?v=HoKDTa5jHvg)

## Common Problems & Fixes

Making sure 
- [Comprehensive StackOverflow Guide](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)

If you are
- To solve this either set

## Future Improvements & To Do List

Karras scheduler paper - https://arxiv.org/abs/2206.00364

Implementing the learned interpolation for beta

Using attention inside the U-Net