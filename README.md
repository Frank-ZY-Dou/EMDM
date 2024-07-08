# Introduction
Official code for the paper **[EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Motion Generation](https://arxiv.org/abs/2312.02256), Arxiv 2023**.


[[Project Page](https://frank-zy-dou.github.io/projects/EMDM/index.html)][[Paper](https://arxiv.org/abs/2312.02256)][[Video](https://www.youtube.com/watch?si=atUuUP4eLbXQjixc&v=1SyCXbnol_g&feature=youtu.be&ab_channel=FrankZhiyangDou)]
<p align="center">
<img src="assets/fig_teaser.png" alt="Image" width="70%">
</p>
We introduce Efficient Motion Diffusion Model (EMDM) for fast and high-quality human motion generation. Although previous motion diffusion models have shown impressive results, they struggle to achieve fast generation while maintaining high-quality human motions. Motion latent diffusion has been proposed for efficient motion generation. However, effectively learning a latent space can be non-trivial in such a two-stage manner. Meanwhile, accelerating motion sampling by increasing the step size, e.g., DDIM, typically leads to a decline in motion quality due to the inapproximation of complex data distributions when naively increasing the step size. In this paper, we propose EMDM that allows for much fewer sample steps for fast motion generation by modeling the complex denoising distribution during multiple sampling steps. Specifically, we develop a Conditional Denoising Diffusion GAN to capture multimodal data distributions conditioned on both control signals, i.e., textual description and denoising time step. By modeling the complex data distribution, a larger sampling step size and fewer steps are achieved during motion synthesis, significantly accelerating the generation process. To effectively capture the human dynamics and reduce undesired artifacts, we employ motion geometric loss during network training, which improves the motion quality and training efficiency. As a result, EMDM achieves a remarkable speed-up at the generation stage while maintaining high-quality motion generation in terms of fidelity and diversity.

# Citation
```angular2html
@article{zhou2023emdm,
  title={EMDM: Efficient Motion Diffusion Model for Fast, High-Quality Motion Generation},
  author={Zhou, Wenyang and Dou, Zhiyang and Cao, Zeyu and Liao, Zhouyingcheng and Wang, Jingbo and Wang, Wenjia and Liu, Yuan and Komura, Taku and Wang, Wenping and Liu, Lingjie},
  journal={arXiv preprint arXiv:2312.02256},
  year={2023}
}
```

# Set up environment
Note that the order of the package installation matters.

```bash
conda create -n emdm python=3.10.13
conda activate emdm
pip3 install torch
pip install "git+https://github.com/openai/CLIP.git"
pip install scipy einops spacy chumpy wandb smplx pandas scikit-learn chardet
pip install matplotlib==3.1.3 --upgrade
pip install numpy==1.23.5 --upgrade
```


# Additional files needed

Follow instructions in [MDM](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file) (Getting started - Setup environment) to obtain the following files: (No need to execute the environment-related commands)

- ./dataset/HumanML3D, KIT-ML, HumanAct12Poses
- ./glove
- ./body_models
- ./assets/actionrecognition
- ./t2m
- ./kit

To download models, see [Google Drive link](https://drive.google.com/drive/folders/1SWsFsHYE4dsXxsX68N1nRYfS4d2PsedZ?usp=sharing) and download the corresponding models.


# Example commands for training
## Training t2m
```bash
CUDA_VISIBLE_DEVICES=0 python train_ddgan.py --dataset {humanml, kit} --exp <experiment name> --lambda_rot_mse 100 --save_dir <directory for saved models>
```

if want to resume from training, add:
```bash
--resume --content_path <path to the content file>
```



## Training a2m
```bash
CUDA_VISIBLE_DEVICES=0 python train_ddgan.py --dataset humanact12 --exp <experiment name> --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 --lambda_rot_mse 1 --save_dir<directory for saved models>
```


## Optional additional keywords examples

**general kwargs**: `--num_timesteps 10 --batch_size 64 --nz 64 --n_mlp 4 --embedding_type positional --use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1e-5 --lazy_reg 15 --transformer_activation selu  --d_type mlp --separate_t_and_action ` 

**model-related kwargs**: `--num_heads 32 --layers 12 --latent_dim 1024`

**save kwargs**: `--save_content_every 60 --save_ckpt_every 10 --log_time_every_x_step 100`

# Command for sampling
## t2m sampling
```
python sample_mdm.py [general kwargs] [model specific kwargs] \
[--input_text ./assets/example_text_prompts.txt] \
--guidance_param 2.5 \
--model_path <path_to_model> \
--dataset {humanml, kit}
```


## a2m sampling
Similar to t2m:

```
python sample_mdm.py
[general kwargs] [model specific kwargs] \
[--action_file ./assets/example_action_names_humanact12.txt] \
--model_path <path_to_model>
```


# Evaluation
## t2m evaluation

To evaluate a single model:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_humanml.py \
--model_path <model_path> \
--eval_mode {fast, wo_mm, mm_short, full} --dataset humanml
```

To evaluate multiple models in the same folder:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_humanml.py \
--block --start 2260 --end 2280 --interval 10 --model_path <model_folder_path> \
--eval_mode {fast, wo_mm, mm_short, full} --dataset humanml
```
for example, this would evaluate model at epochs 2260, 2270, 2280


## a2m evaluation
```bash
CUDA_VISIBLE_DEVICES=1 python eval_humanact12_uestc.py --dataset humanact12 --cond_mask_prob 0 --eval_mode full --guidance_param 1 --model_path <model path>
```


## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.