from absl import app, flags
import enum
import numpy as np
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.simple_unet import SimpleUnet
from data_loader import load_transformed_dataset
from forward import ForwardDiffusion

FLAGS = flags.FLAGS
_MODE = flags.DEFINE_enum(name = 'mode', default = None, enum_values = ['train', 'eval'], help = "Running mode")
#_EPOCHS = flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training.')
# flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training.')

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_ROOT_DIR, 'data')
_OUTPUT_DIR = os.path.join(_ROOT_DIR, 'output')
_MODEL_PATH = os.path.join(_OUTPUT_DIR, ".model.pt")
_IMG_SIZE = 64
_BATCH_SIZE = 128
#T = 300
_EPOCHS = 20
T = 10

def _train():
    print("Start train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUnet()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)
    data = load_transformed_dataset(_IMG_SIZE, _DATA_DIR)
    dataloader = DataLoader(data, batch_size=_BATCH_SIZE, shuffle=True, drop_last=True)

    forward_diffusion = ForwardDiffusion(device)
    forward_diffusion.pre_calculate_param(T)

    for epoch in range(_EPOCHS):
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

def _train_mnist():
    # hardcoding these here
    batch_size = 256
    n_T = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = False
    save_dir = './data/diffusion_outputs10/'
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(_EPOCHS):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/_EPOCHS)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        with torch.no_grad():
            n_sample = 4*n_classes
            for w_i, w in enumerate(ws_test):
                x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample/n_classes)):
                        try: 
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k+(j*n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all*-1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep%5==0 or ep == int(_EPOCHS-1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
        # optionally save model
        if save_model and ep == int(_EPOCHS-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

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
