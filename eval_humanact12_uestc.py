"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
import torch
import re

from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from eval.a2m.tools import save_metrics
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
import argparse
from parsing_utils import ddgan_parser_add_argument


def evaluate(args, model, diffusion, data):
    scale = None
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
        scale = {
            'action': torch.ones(args.batch_size) * args.guidance_param,
        }
    model.to(dist_util.dev())
    model.eval()  # disable random masking


    folder, ckpt_name = os.path.split(args.model_path)
    if args.dataset == "humanact12":
        from eval.a2m.gru_eval import evaluate
        eval_results = evaluate(args, model, diffusion, data)
    elif args.dataset == "uestc":
        from eval.a2m.stgcn_eval import evaluate
        eval_results = evaluate(args, model, diffusion, data)
    else:
        raise NotImplementedError("This dataset is not supported.")

    # save results
    iter = int(re.findall('\d+', ckpt_name)[0])
    scale = 1 if scale is None else scale['action'][0].item()
    scale = str(scale).replace('.', 'p')
    metricname = "evaluation_results_iter{}_samp{}_scale{}_a2m.yaml".format(iter, args.num_samples, scale)
    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, eval_results)

    return eval_results


def eval_humanact12_uestc(args):
    # setattr(args, 'cond_mode', 'action')
    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    print(f'Eval mode [{args.eval_mode}]')
    assert args.eval_mode in ['debug', 'full', 'fast'], f'eval_mode {args.eval_mode} is not supported for dataset {args.dataset}'
    if args.eval_mode == 'debug':
        args.num_samples = 10
        args.num_seeds = 2
    elif args.eval_mode == 'full':
        args.num_samples = 1000
        args.num_seeds = 20
    else: # fast
        args.num_samples = 1000
        args.num_seeds = 1

    data_loader = get_dataset_loader(name=args.dataset, num_frames=60, batch_size=args.batch_size,)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data_loader)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')['netG_dict']
    load_model_wo_clip(model, state_dict)

    eval_results = evaluate(args, model, diffusion, data_loader.dataset)

    fid_to_print = {k : sum([float(vv) for vv in v])/len(v) for k, v in eval_results['feats'].items() if 'fid' in k and 'gen' in k}
    print(fid_to_print)
    return eval_results, fid_to_print

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

if __name__ == '__main__':
    import numpy as np
    parser = argparse.ArgumentParser('ddgan parameters')
    ddgan_parser_add_argument(parser)
    parser.add_argument('--block', action='store_true',
                        help='if true, eval a sequence of nodes instead of a single one, and need to provide start, \
                        end and interval; also model_path should be a folder')
    parser.add_argument('--start', type=int, default=0, help='start epoch')
    parser.add_argument('--end', type=int, default=100, help='end epoch')
    parser.add_argument('--interval', type=int, default=10, help='interval between epochs')
    args = evaluation_parser(parser)

    print('args:', args)

    # batch = False
    # batch = True
    if not args.block:  # single
        # setattr(args, 'model_path',
        #         f'/home/andy/Documents/EMDM related files/emdm ablation models/humanact uncond/000007000.pth')
        result, fid = eval_humanact12_uestc(args)
        print('eval_result:', result)
        print('fid:', fid)

        result = result['feats']
        for key, result in result.items():
            if 'gt' not in key:
                result = [float(v) for v in result]
                #print(key, np.mean(result), np.std(result))
                mean, ci = get_metric_statistics(result, args.num_seeds)
                # print(f'{key}: {np.mean(result):.3f} +- {np.std(result):.3f}')
                print(f'{key}: {mean:.3f} +- {ci:.3f}')
    else:
        results = dict()
        base_path = args.model_path
        for i in range(args.start, args.end+1, args.interval):
            # setattr(args, 'model_path', f'./saved_info/dd_gan/uestc BBT/{i:09d}.pth')
            setattr(args, 'model_path', f'{base_path}/{i:09d}.pth')
            # args.eval_mode = 'fast'
            # args.num_timesteps = 10
            result, fid = eval_humanact12_uestc(args)
            print(i, result)

            result = result['feats']
            for key, result in result.items():
                if 'gt' not in key:
                    result = [float(v) for v in result]
                    if key not in results:
                        results[key] = []
                    results[key].append((i, sum(result) / len(result)))

        for key, value in results.items():
            print(key)
            for i, v in value:
                print(f'{v:.4f}')
            print()
