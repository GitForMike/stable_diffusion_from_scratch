# stable_diffusion_from_scratch


## How to train
```console
$ python3 train.py  --config 'config/unconditional_ddpm_with_dummy_eps.yaml'
```
The trained state dict of model will be stored in output folder.

## How to inference
The inferenced image will be stored in output folder.
```console
$ 
```


## Reference

### DDPM
https://github.com/explainingai-code/DDPM-Pytorch/tree/main

https://github.com/cloneofsimo/minDiffusion

https://github.com/TeaPearce/Conditional_Diffusion_MNIST/tree/main

https://jang-inspiration.com/ddpm-1

https://xoft.tistory.com/32

### LDM
https://github.com/explainingai-code/StableDiffusion-PyTorch/tree/main

https://arxiv.org/abs/2112.10752

https://github.com/CompVis/latent-diffusion

## TODO

- [ ] Unconditional DDPM with dummy eps model
- [ ] Unconditional DDPM with UNET eps model
- [ ] Conditional DDPM
- [ ] Unconditional LDM
- [ ] Conditional LDM