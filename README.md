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

- [OpenAI CLIP Explanation](https://openai.com/index/clip/)

- [CLIP Paper - Learning Transferable Visual Models from Language Supervision](https://arxiv.org/pdf/2103.00020)

### Other sources

I would highly recommend reading these to get a good understanding into.

3Blue1Brown - [Mathematics behind transformers - Chapter 5/6](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=7kUJ3D5-B24sOq7j)

## Common Problems & Fixes

Making sure 
- [Comprehensive StackOverflow Guide](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)

If you are
- To solve this either set

## Future Improvements & To Do List

Karras scheduler paper - https://arxiv.org/abs/2206.00364
Implement the custom schedulers