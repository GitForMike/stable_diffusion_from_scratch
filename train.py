from absl import app, flags
import enum
import numpy as np
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yaml

FLAGS = flags.FLAGS
_CONFIG = flags.DEFINE_string(name = 'config', default = None, help = "config_path")

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_ROOT_DIR, 'data')
_OUTPUT_DIR = os.path.join(_ROOT_DIR, 'output')


def _read_config():
    config = None
    with open(_CONFIG.value, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as err:
            print(err)
    print(config)
    return config

def main(argv):
    config = _read_config()
    return

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(_OUTPUT_DIR, f"{_MODEL.value}.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    if _MODEL.value == "ddpm":
        model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    elif _MODEL.value == "cond":
        model = ConditionalDiffusion(nn_model=ContextUnet(in_channels=1, n_feat=128, n_classes=10), betas=(1e-4, 0.02), n_T=400, device=device, drop_prob=0.1)       
    model.to(device)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )

    dataset = MNIST(
        "./data",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=20)
    optim = torch.optim.Adam(model.parameters(), lr=_LRATE)

    for i in range(_EPOCHS):
        model.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = _LRATE*(1-i/_EPOCHS)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            if _MODEL.value == "ddpm":
                loss = model(x)
            elif _MODEL.value == "cond":
                loss = model(x,c)

            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

        # save model
        torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    app.run(main)
