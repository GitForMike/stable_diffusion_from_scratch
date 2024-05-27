from absl import app, flags
import torch
import os
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from models.dummy_eps_model import DummyEpsModel
from models.unet import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.file import read_yaml_file

FLAGS = flags.FLAGS
_CONFIG = flags.DEFINE_string(name = 'config', default = None, help = "config_path")

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_ROOT_DIR, 'data')
_OUTPUT_DIR = os.path.join(_ROOT_DIR, 'output')

def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_yaml_file(_CONFIG.value)
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    model_path = os.path.join(_OUTPUT_DIR, f"{os.path.basename(_CONFIG.value).split('.')[0]}.pth")

    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])

    model = None
    if model_config['name'] == "dummy_eps_model":
        model = DummyEpsModel(model_config)
    elif model_config['name'] == "unet":
        model = Unet(model_config)
    model = model.to(device)
   
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        xt = torch.randn((train_config['num_samples'],
                    model_config['im_channels'],
                    model_config['im_size'],
                    model_config['im_size'])).to(device)
        grid = None
        for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
            # Get prediction of noise
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

            # Use scheduler to get x0 and xt-1
            xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

            # Save x0
            ims = torch.clamp(xt, -1., 1.).detach().cpu()
            ims = (ims + 1) / 2
            #grid = make_grid(ims, nrow=train_config['num_grid_rows'])
            grid = make_grid(ims, nrow=4)
            #img = transforms.ToPILImage()(grid)
            #if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            #    os.mkdir(os.path.join(train_config['task_name'], 'samples'))
            #img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
            #img.close()
        img = transforms.ToPILImage()(grid)
        img.save(os.path.join(_OUTPUT_DIR, f"{os.path.basename(_CONFIG.value).split('.')[0]}_{i}.png"))
        img.close()

    print("End eval")
        

if __name__ == '__main__':
    app.run(main)
