import torch


class LinearNoiseScheduler:
    """
    Class for the linear noise scheduler.
    """

    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self._pre_calculate_params()

    def _pre_calculate_params(self):
        """
        Pre-calculates params to be used for forward diffusion process.
        
        """
        # Linear noise schedule.
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)


    def add_noise(self, original, noise, t):
        """
        Adds noise on timestep t if on when original image is given.
        """
        device = original.device
        batch_size = original.shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(device)[t].reshape(batch_size)

        # Reshape till (B,) becomes (B,1,1,1) if image is (B,C,H,W)
        # TODO: Check why we need this??
        for _ in range(len(original.shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return (sqrt_alpha_cum_prod.to(original.device) * original
                + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise)

if __name__ == '__main__':
    scheduler = LinearNoiseScheduler(100, 0.002,0.1)
    print(scheduler.sqrt_alpha_cum_prod[0].reshape(scheduler.sqrt_alpha_cum_prod.shape[0]))