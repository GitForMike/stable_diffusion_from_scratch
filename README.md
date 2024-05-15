# stable_diffusion_from_scratch


## How to train
```bash
$ python3 train_ddpm.py --config 'config/unconditional_ddpm_with_dummy_eps.yaml'
$ python3 train_ddpm.py --config 'config/unconditional_ddpm_with_unet.yaml'

$ python3 train_vae.py --config 'config/unconditional_ldm.yaml'
```
The trained state dict of model will be stored in output folder.

## How to inference
The inferenced image will be stored in output folder.
```bash
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

- [x] Unconditional DDPM with dummy eps model
- [x] Unconditional DDPM with UNET eps model
- [ ] Conditional DDPM
- [ ] Unconditional LDM
- [ ] Conditional LDM


## Data Preparation
### Mnist

For setting up the mnist dataset follow - https://github.com/explainingai-code/Pytorch-VAE#data-preparation

Ensure directory structure is following
```
StableDiffusion-PyTorch
    -> data
        -> mnist
            -> train
                -> images
                    -> *.png
            -> test
                -> images
                    -> *.png
```

### CelebHQ 
#### Unconditional
For setting up on CelebHQ for unconditional, simply download the images from the official repo of CelebMASK HQ [here](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file).

Ensure directory structure is the following
```
StableDiffusion-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg

```
#### Mask Conditional
For CelebHQ for mask conditional LDM additionally do the following:

Ensure directory structure is the following
```
StableDiffusion-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg
            -> CelebAMask-HQ-mask-anno
                -> 0/1/2/3.../14
                    -> *.png
            
```

* Run `python -m utils.create_celeb_mask` from repo root to create the mask images from mask annotations

Ensure directory structure is the following
```
StableDiffusion-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg
            -> CelebAMask-HQ-mask-anno
                -> 0/1/2/3.../14
                    -> *.png
            -> CelebAMask-HQ-mask
                  -> *.png
```

#### Text Conditional
For CelebHQ for text conditional LDM additionally do the following:
* The repo uses captions collected as part of this repo - https://github.com/IIGROUP/MM-CelebA-HQ-Dataset?tab=readme-ov-file 
* Download the captions from the `text` link provided in the repo - https://github.com/IIGROUP/MM-CelebA-HQ-Dataset?tab=readme-ov-file#overview
* This will download a `celeba-captions` folder, simply move this inside the `data/CelebAMask-HQ` folder as that is where the dataset class expects it to be.

Ensure directory structure is the following
```
StableDiffusion-PyTorch
    -> data
        -> CelebAMask-HQ
            -> CelebA-HQ-img
                -> *.jpg
            -> CelebAMask-HQ-mask-anno
                -> 0/1/2/3.../14
                    -> *.png
            -> CelebAMask-HQ-mask
                -> *.png
            -> celeba-caption
                -> *.txt
```
---
