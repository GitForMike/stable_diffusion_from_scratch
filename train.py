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


#from model.simple_unet import SimpleUnet
from models.ddpm import DDPM, DummyEpsModel
from models.conditional_diffusion import ConditionalDiffusion, ContextUnet
#from data_loader import load_transformed_dataset
#from forward import ForwardDiffusion

FLAGS = flags.FLAGS
_MODEL = flags.DEFINE_enum(name = 'model', default = None, enum_values = ['ddpm', 'cond'], help = "model")
#_MODE = flags.DEFINE_enum(name = 'mode', default = None, enum_values = ['train', 'eval'], help = "Running mode")
#_EPOCHS = flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training.')
# flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training.')

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_ROOT_DIR, 'data')
_OUTPUT_DIR = os.path.join(_ROOT_DIR, 'output')
_IMG_SIZE = 64
_BATCH_SIZE = 128
_EPOCHS = 20
_LRATE = 1e-4

def main(argv):
    del argv
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
