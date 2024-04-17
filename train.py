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

from model.dummy_eps_model import DummyEpsModel
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _read_config()
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    # Create output directories
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(_OUTPUT_DIR, f"{_CONFIG.value.split('.')[0]}.pth")

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])



    # Create the dataset
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0))]
    )
    dataset = MNIST(
        dataset_config['im_path'],
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=20)

    # Create the model
    model = None
    if model_config['name'] == "dummy_eps_model":
        model = DummyEpsModel(model_config)
    model.to(device)
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=train_config["lr"])

    # TODO: from here
    for i in range(_EPOCHS):
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
