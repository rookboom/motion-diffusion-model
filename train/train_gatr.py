# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import sys
sys.path.insert(0, '/root/repos/motion-diffusion-model/model')
import torch
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_gaussian_diffusion, get_cond_mode
from train.train_platforms import TensorboardPlatform, NoPlatform  # required for the eval operation
from model.gdm import GDM

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Copy the training script to the save directory so we know how to repeat the run:
    # The training script is called train.sh, which calls this file train_gatr.py
    os.system('cp {} {}'.format("train.sh", args.save_dir))
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),)

    print("creating model and diffusion...")
    model = create_gatr_model(args, data)
    model = torch.compile(model)
    diffusion = create_gaussian_diffusion(args)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    # breakpoint()
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


def create_gatr_model(args, data):
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1
    cond_mode = get_cond_mode(args)

    return GDM(
        num_actions,
        num_heads=args.num_heads, 
        hidden_mv_channels=args.hidden_mv_channels, 
        hidden_s_channels=args.hidden_s_channels,
        dataset=args.dataset,
        cond_mode=cond_mode,
        num_blocks=args.layers,
        latent_dim=args.latent_dim,
        dropout=0.1,
        cond_mask_prob=args.cond_mask_prob,
        pos_embed_max_len=args.pos_embed_max_len
        )

if __name__ == "__main__":
    main()
