import numpy as np
import os
import shutil
import torch



def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    # print('var schedule', var)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    # print('sigma schedule', sigmas, args)
    return sigmas, a_s, betas


class Diffusion_Coefficients():
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise

    return x_t


def q_sample_pairs(coeff, x_start, t):
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise

    return x_t, x_t_plus_one


# %% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        # self.posterior_mean_coef1 = torch.tensor([1.,        0.73481859, 0.55341782, 0.44587667, 0.37721745, 0.33035602,
        #             0.29597059, 0.26714337, 0.2337403,  0.15506361]).cuda()
        # self.posterior_mean_coef2 = torch.tensor([0. ,        0.26490377, 0.4448114,  0.54830975, 0.60825173, 0.63787432,
        #             0.63861734, 0.5985179,  0.469853,   0.03086167]).cuda()
        # self.posterior_log_variance_clipped = torch.tensor([-3.87284789, -3.87284789, -2.82796961, -2.23393557, -1.80024125, -1.43572568,
        #             -1.09602252, -0.75004728, -0.37032326, -0.02536308]).cuda()
        #
        print(f'posterior_mean_coef1: {self.posterior_mean_coef1}')
        print(f'posterior_mean_coef2: {self.posterior_mean_coef2}')
        print(f'posterior_log_variance_clipped: {self.posterior_log_variance_clipped}')


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, T, opt, y):
    x = x_init
    if 'text' in y: # t2m, can be sped up by precomputing text encodings
        from model.cfg_sampler import ClassifierFreeSampleModel
        with torch.no_grad():
            text_encode = generator.model.encode_text(y['text'])
            for i in reversed(range(n_time)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

                t_time = t
                latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
                if isinstance(generator, ClassifierFreeSampleModel):
                    x_0 = generator(x, t_time, latent_z, y, text_encode)
                else:
                    x_0 = generator(x, t_time, latent_z, y)
                x_new = sample_posterior(coefficients, x_0, x, t)
                x = x_new.detach()

        return x
    else: # a2m
        with torch.no_grad():
            for i in reversed(range(n_time)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

                t_time = t
                latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
                x_0 = generator(x, t_time, latent_z, y)
                x_new = sample_posterior(coefficients, x_0, x, t)
                x = x_new.detach()

        return x