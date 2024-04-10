from absl import app, flags
import enum
import numpy as np
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import SimpleUnet
from data_loader import load_transformed_dataset
from forward import ForwardDiffusion

FLAGS = flags.FLAGS
_MODE = flags.DEFINE_enum(name = 'mode', default = None, enum_values = ['train', 'eval'], help = "Running mode")
_EPOCHS = flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training.')
# flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training.')

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_ROOT_DIR, 'data')
_OUTPUT_DIR = os.path.join(_ROOT_DIR, 'output')
_MODEL_PATH = os.path.join(_OUTPUT_DIR, ".model.pt")
_IMG_SIZE = 64
_BATCH_SIZE = 128
#T = 300
T = 10

def _train():
    print("Start train")
    model = SimpleUnet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    data = load_transformed_dataset(_IMG_SIZE, _DATA_DIR)
    dataloader = DataLoader(data, batch_size=_BATCH_SIZE, shuffle=True, drop_last=True)

    forward_diffusion = ForwardDiffusion(device)
    forward_diffusion.pre_calculate_param(T)

    for epoch in range(_EPOCHS.value):
        print("help1")
        for step, batch in enumerate(dataloader):
            print("help2")
            optimizer.zero_grad()

            t = torch.randint(0, T, (_BATCH_SIZE,), device=device).long()

            x_noisy, noise = forward_diffusion.forward_diffusion_sample(batch[0], t)
            noise_pred = model(x_noisy, t)
            loss = F.l1_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()

            '''
            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image()
            '''

    # Save trained model
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    torch.save(model, _MODEL_PATH)
    print("End train")

def _eval():
    print("Start eval")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(_MODEL_PATH)
    model.eval()

    data = torch.randn(_BATCH_SIZE,3,64,64)
    t = torch.randint(0,T, (_BATCH_SIZE,), device=device).long()
    output = model.forward(data, t)
    print(output)
    print(type(output))
    show_tensor_image(output.detach().numpy())

    print("End eval")

def main(argv):
    del argv

    if _MODE.value == 'train':
        _train()
    elif _MODE.value == 'eval':
        _eval()
    else:
        raise ValueError(f'Invalid mode: {_MODE.value}')
        

if __name__ == '__main__':
    app.run(main)
