# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

# from . import up_or_down_sampling
from . import dense_layer
from . import layers

dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


# %%
class DownConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1,
            t_emb_dim=128,
            downsample=False,
            act=nn.LeakyReLU(0.2),
            fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.fir_kernel = fir_kernel
        self.downsample = downsample

        self.conv1 = nn.Sequential(
            conv2d(in_channel, out_channel, kernel_size, padding=padding),
        )

        self.conv2 = nn.Sequential(
            conv2d(out_channel, out_channel, kernel_size, padding=padding, init_scale=0.)
        )
        self.dense_t1 = dense(t_emb_dim, out_channel)

        self.act = act

        self.skip = nn.Sequential(
            conv2d(in_channel, out_channel, 1, padding=0, bias=False),
        )

    def forward(self, input, t_emb):
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]

        out = self.act(out)

        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)

        return out


class Discriminator_small(nn.Module):
    """A time-dependent discriminator for small images (CIFAR10, StackMNIST)."""

    def __init__(self, nc=3, ngf=64, t_emb_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        # Encoding layers where the resolution decreases
        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 2, t_emb_dim=t_emb_dim, act=act)

        self.conv2 = DownConvBlock(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv3 = DownConvBlock(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv4 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1, init_scale=0.)
        self.end_linear = dense(ngf * 8, 1)

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        h0 = self.start_conv(input_x)
        h1 = self.conv1(h0, t_embed)

        h2 = self.conv2(h1, t_embed)

        h3 = self.conv3(h2, t_embed)

        out = self.conv4(h3, t_embed)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)

        return out


class Discriminator_large(nn.Module):
    """A time-dependent discriminator for large images (CelebA, LSUN)."""

    def __init__(self, nc=1, ngf=32, t_emb_dim=128, act=nn.LeakyReLU(0.2)):
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.act = act

        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )

        self.start_conv = conv2d(nc, ngf * 2, 1, padding=0)
        self.conv1 = DownConvBlock(ngf * 2, ngf * 4, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv2 = DownConvBlock(ngf * 4, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv3 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.conv4 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv5 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)
        self.conv6 = DownConvBlock(ngf * 8, ngf * 8, t_emb_dim=t_emb_dim, downsample=True, act=act)

        self.final_conv = conv2d(ngf * 8 + 1, ngf * 8, 3, padding=1)
        self.end_linear = dense(ngf * 8, 1)

        self.stddev_group = 4
        self.stddev_feat = 1

    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))

        input_x = torch.cat((x, x_t), dim=1)

        h = self.start_conv(input_x)
        h = self.conv1(h, t_embed)

        h = self.conv2(h, t_embed)

        h = self.conv3(h, t_embed)
        h = self.conv4(h, t_embed)
        h = self.conv5(h, t_embed)

        out = self.conv6(h, t_embed)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = self.act(out)

        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)

        return out


class DiscriminatorMDM_MLP(nn.Module):
    def __init__(self, args: dict):
        super(DiscriminatorMDM_MLP, self).__init__()

        act = nn.LeakyReLU(0.2)
        self.t_embed = TimestepEmbedding(
            embedding_dim=128,
            hidden_dim=128,
            output_dim=128,
            act=act,
        )

        dataset = args.get('dataset', None)
        if dataset is None:
            raise ValueError('dataset not specified')
        self.dataset = dataset
        dataset_param = dict()
        if dataset == 'humanact12':
            dataset_param['input_size'] = [25, 6, 60]
            dataset_param['num_extra_features'] = 12
        elif dataset == 'humanml':
            dataset_param['input_size'] = [263, 1, 196]
            dataset_param['num_extra_features'] = 512
        elif dataset == 'uestc':
            dataset_param['input_size'] = [25, 6, 60]
            dataset_param['num_extra_features'] = 40
        elif dataset == 'kit':
            dataset_param['input_size'] = [251, 1, 196]
            dataset_param['num_extra_features'] = 512
        else:
            raise NotImplementedError(f"Dataset not supported: {dataset}")
        print(args)

        self.d_no_cond = args.get('d_no_cond', False)


        # Fully connected layers
        if self.d_no_cond:
            print('d_no_cond!')
            self.fc1 = nn.Linear(np.product(dataset_param['input_size']) * 2 + 128, 1024)
        else:
            self.fc1 = nn.Linear(np.product(dataset_param['input_size']) * 2 +
                                 128 + # time emb
                                 dataset_param['num_extra_features']
                                 , 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 64)
        self.fc7 = nn.Linear(64, 1)


        # Activation and pooling layers
        # self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.gn1 = nn.GroupNorm(32, 1024)
        self.gn2 = nn.GroupNorm(16, 512)


    def forward(self, x_t, t, x_tp1, y, **kwargs):
        x_t = x_t.reshape(x_t.size(0), -1)  # Flatten the tensor
        x_tp1 = x_tp1.view(x_tp1.size(0), -1)  # Flatten the tensor
        t = self.t_embed(t)

        if not self.d_no_cond:
            if self.dataset == 'humanact12':
                # one-hot encode y with 12 classes
                y = torch.nn.functional.one_hot(y['action'], num_classes=12).float().squeeze(1).cuda()
                # print('one-hot y: ', y.shape, y)
            elif self.dataset == 'humanml' or self.dataset == 'kit':
                # print('d, y:', y)
                y = kwargs['encode_function'](y['text'])
                # print(f'y shape: {y.shape}')
            elif self.dataset == 'uestc':
                y = torch.nn.functional.one_hot(y['action'], num_classes=40).float().squeeze(1).cuda()
            else:
                raise "Dataset not supported"

            x = torch.cat((x_t, x_tp1, t, y), dim=1)
        else:
            x = torch.cat((x_t, x_tp1, t), dim=1)

        x = self.fc1(x)
        x = self.selu(x)

        x = self.fc2(x)
        x = self.gn1(x)
        x = self.selu(x)

        x = self.fc3(x)
        x = self.selu(x)

        x = self.fc4(x)
        x = self.gn2(x)
        x = self.selu(x)

        x = self.fc5(x)
        x = self.selu(x)

        x = self.fc6(x)
        x = self.selu(x)

        x = self.fc7(x)

        # reduce to 1D
        x = x.squeeze(1)

        return x
