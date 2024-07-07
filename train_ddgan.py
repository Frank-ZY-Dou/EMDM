# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse

import torch.nn.functional as F
import torch.optim as optim

from data_loaders.get_data import get_dataset_loader
from utils.model_util import get_model_args
from model.mdm_discriminator import MDM_Discriminator
from utils.parser_util import train_args as mdm_args_parse
from utils.model_util import create_model_and_diffusion
import time
from score_sde.models.discriminator import DiscriminatorMDM_MLP

from diffusion.sampling import *

import wandb
from timer import Timer
from parsing_utils import ddgan_parser_add_argument


def train(args):
    from EMA import EMA

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:{}'.format(0))

    batch_size = args.batch_size
    nz = args.nz  # latent dimension

    data_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=60)

    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)

    netG, diffusionG = create_model_and_diffusion(args, data_loader)
    netG.to('cuda')  # netG = netG.to(device) would make netG None

    if args.d_type == 'mlp':
        netD = DiscriminatorMDM_MLP(get_model_args(args, data_loader))
        netD = netD.to(device)
    elif args.d_type == 'encoder':
        netD = MDM_Discriminator(**get_model_args(args, data_loader))
        netD.to(device)
    else:
        raise NotImplementedError(f'unknown d_type: {args.d_type}')

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
        print('EMA state:', optimizerG.state_dict())

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    exp = args.exp
    parent_dir = args.save_dir

    exp_path = os.path.join(parent_dir, exp)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    if args.resume:
        checkpoint_file = args.content_path
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch

        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    timer = Timer()

    def wandb_log(*args_log, **kwargs):
        # stop timer and to the log to wandb then resume
        timer.pause()
        if not args.disable_wandb:
            wandb.log(*args_log, **kwargs)
        timer.start()

    timer.start()

    for epoch in range(init_epoch, args.num_epoch + 1):
        wandb_log({"epoch": epoch}, step=global_step)
        t0 = time.time()
        for iteration, (x, y) in enumerate(data_loader):
            y = y['y']
            y['mask'] = y['mask'].to(device)

            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            # sample from p(x_0)
            real_data = x.to(device, non_blocking=True)

            # sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)

            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            x_t.requires_grad = True

            # train with real
            D_real = netD(x_t, t, x_tp1.detach(), y, encode_function=netG.encode_text).view(-1)
            # check if nan
            if torch.isnan(D_real).any():
                print('D_real nan!!')
                import sys
                sys.exit(0)

            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()
            wandb_log({"D_real": errD_real.item()}, step=global_step)

            errD_real.backward(retain_graph=True)

            if args.lazy_reg is None:
                print('lazy reg')
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True
                )[0]
                grad_penalty = (
                        grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()
                if torch.isnan(grad_penalty).any():
                    print('grad_penalty nan')
                    import sys
                    sys.exit(0)

                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            else:
                if global_step % args.lazy_reg == 0:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]

                    grad_penalty = (
                            grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()

                    del grad_penalty, grad_real

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)

            x_0_predict = netG(x_tp1.detach(), t, latent_z, y)
            if torch.isnan(x_0_predict).any():
                print('x_0_predict nan')
                import sys
                sys.exit(0)
            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach(), y, encode_function=netG.encode_text).view(-1)
            if torch.isnan(output).any():
                print('output1 nan')
                import sys
                sys.exit(0)

            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()

            wandb_log({"D_fake": errD_fake.item()}, step=global_step)

            errD_fake.backward()

            # Update D
            optimizerD.step()
            # check nan in weights
            for p in netD.parameters():
                if torch.isnan(p).any():
                    print('netD nan')
                    import sys
                    sys.exit(0)

            del D_real, errD_real, errD_fake, x_t, x_tp1, x_pos_sample, output

            # update G
            # first calculate mdm geometric loss
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z, y)
            if torch.isnan(x_0_predict).any():
                print('x_0_predict nan')
                import sys
                sys.exit(0)

            terms = diffusionG.training_losses(netG, x_tp1.detach(), x.cuda(), t, x_0_predict, model_kwargs={'y': y},
                                               dataset=data_loader.dataset)
            terms['loss'].mean().backward()

            for k in terms:
                wandb_log({f"terms-{k}": terms[k].mean().item()}, step=global_step)

            # then calculate with discriminator
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z, y)

            x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

            output = netD(x_pos_sample, t, x_tp1.detach(), y, encode_function=netG.encode_text).view(-1)
            if torch.isnan(output).any():
                print('output2 nan')
                import sys
                sys.exit(0)

            errG = F.softplus(-output)
            errG = errG.mean()

            wandb_log({"G": errG.item()}, step=global_step)

            if not args.disable_gan:  # only backprop if not disabled
                errG.backward()

            g_prev = netG.state_dict()
            optimizerG.step()
            # check nan in weights
            for p in netG.parameters():
                if torch.isnan(p).any():
                    print('netG nan')
                    print('prev:', g_prev)
                    print('current:', netG.state_dict())
                    import sys
                    sys.exit(0)
            del g_prev

            global_step += 1
            if epoch == 1 and iteration == 0:
                print(f'global steps per epoch: {global_step}')

            del x_t, x_tp1, latent_z, x_0_predict, x_pos_sample, output, errG, terms, t

            timer.pause()
            if global_step % args.log_time_every_x_step == 0:
                # also log timer time
                wandb_log({"time": timer.current_time()}, step=global_step)
            timer.start()

        if not args.no_lr_decay:
            schedulerG.step()
            schedulerD.step()

        if epoch % args.save_content_every == 0:
            timer.pause()
            print('saving content...')

            content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                       'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                       'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                       'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}

            torch.save(content, os.path.join(exp_path, 'content_{}.pth'.format(epoch)))
        if epoch % args.save_ckpt_every == 0:  # or epoch % args.eval_step == 0:
            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
            print('ready to save to:', os.path.join(exp_path, f'{epoch:09d}.pth'))
            model_path = os.path.join(exp_path, f'{epoch:09d}.pth')
            torch.save({'epoch': epoch, 'netG_dict': netG.state_dict(),  # 'netD_dict': netD.state_dict(),
                        'global_step': global_step}, model_path)

            if args.use_ema:
                optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

            timer.start()

        if epoch < 10:  # print time for first 10 epochs
            print(f'epoch {epoch} time: {time.time() - t0}')


# %%
if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('ddgan parameters')
    ddgan_parser_add_argument(parser)

    args = mdm_args_parse(parser)

    # auto set saving parameters
    if args.save_ckpt_every == 0:
        step = {'humanact12': 500, 'uestc': 15, 'humanml': 10, 'kit': 100}
        setattr(args, 'save_ckpt_every', step[args.dataset])
    if args.save_content_every == 0:
        step = {'humanact12': 1000, 'uestc': 50, 'humanml': 30, 'kit': 300}
        setattr(args, 'save_content_every', step[args.dataset])

    if not args.disable_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=f'emdm-{args.dataset}',

            # track hyperparameters and run metadata
            config=args,

            name=args.exp
        )

    train(args)
