from absl import app, flags
import torch
import os
from torchvision.utils import save_image, make_grid

from models.ddpm import DDPM, DummyEpsModel
from models.conditional_diffusion import ConditionalDiffusion, ContextUnet

FLAGS = flags.FLAGS
_MODEL = flags.DEFINE_enum(name = 'model', default = None, enum_values = ['ddpm', 'cond'], help = "model")

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_ROOT_DIR, 'data')
_OUTPUT_DIR = os.path.join(_ROOT_DIR, 'output')



def cond_inference(model, device):
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
    n_classes = 10
    number = 5 # Any number from 0 ~ 9

    n_sample = 4*n_classes
    for w_i, w in enumerate(ws_test):
        x_gen, x_gen_store = model.sample(n_sample, (1, 28, 28), device, guide_w=w, number_to_inference = number)
        grid = make_grid(x_gen*-1 + 1, nrow=10)
        save_image(grid, os.path.join(_OUTPUT_DIR, f"image_w{w}.png"))
        print('saved image at ' + os.path.join(_OUTPUT_DIR, f"image_w{w}.png"))


def main(argv):
    print("Start eval")
    model_path = os.path.join(_OUTPUT_DIR, f"{_MODEL.value}.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = None
    if _MODEL.value == "ddpm":
        model = DDPM(eps_model=DummyEpsModel(1), betas=(1e-4, 0.02), n_T=1000)
    elif _MODEL.value == "cond":
        model = ConditionalDiffusion(nn_model=ContextUnet(in_channels=1, n_feat=128, n_classes=10), betas=(1e-4, 0.02), n_T=400, device=device, drop_prob=0.1)       

    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        if _MODEL.value == "ddpm":
            xh = model.sample(16, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, os.path.join(_OUTPUT_DIR, f"test_result.png"))
            print('saved image at ' + os.path.join(_OUTPUT_DIR, f"test_result.png"))
        elif _MODEL.value == "cond":
            cond_inference(model, device)
     

    print("End eval")
        

if __name__ == '__main__':
    app.run(main)
