def ddgan_parser_add_argument(parser):
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--log_time_every_x_step', type=int, default=200,
                        help='log time every ? steps')
    parser.add_argument('--eval_step', type=int, default=2,
                        help='evaluate time every ? steps')
    parser.add_argument('--d_no_cond', action='store_true', default=False,
                        help='do not feed y to discriminator')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=4,
                        help='number of mlp layers for z')


    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--nz', type=int, default=64)
    parser.add_argument('--num_timesteps', type=int, default=10)

    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1_000_000)

    parser.add_argument('--lr_g', type=float, default=2e-5, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1.25e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.02, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=15,
                        help='lazy regularization.')

    parser.add_argument('--save_content_every', type=int, default=0, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=0, help='save ckpt every x epochs')

    parser.add_argument('--content_path', type=str, help='specify where to load contents to resume training')

    parser.add_argument('--transformer_activation', type=str, default='selu',
                        help='transformer activation (gelu / selu...)')

    parser.add_argument('--exp', help='name of experiment')
    parser.add_argument('--d_type', choices=['mlp', 'encoder'], default="mlp", help='discriminator architecture')
    parser.add_argument('--separate_t_and_action', action='store_true', default=True,
                        help='separate t and action/text when fed to encoder')
    parser.add_argument('--disable_gan', action='store_true', help='disable discriminator loss and z_emb')

    parser.add_argument("--lambda_rot_mse", default=1.0, type=float, help="rot_mse loss")
    parser.add_argument("--disable_wandb", action='store_true', help="disable syncing to wandb")