import os
import glob
import argparse

import  colossalai
import torch
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.lr_scheduler import LinearWarmupLR
from colossalai.trainer import Trainer, hooks

from imagenet_dataloader import DaliDataloader
from myhooks import TotalBatchsizeHook
from model import Widenet

# Training settings
parser = argparse.ArgumentParser(description='WideNet (Colossal-AI implementation) training on GPU devices',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log_dir', default='./log/',
                    help='log file saved place')
parser.add_argument('--checkpoint-dir', default='./checkpoint',
                    help='checkpoint file saved place')
parser.add_argument('--tpu', default='',
                    help='TPU name')
parser.add_argument('--data_set', choices=['Imagenet', 'Cifar10', 'Cifar100'],
                    default='Imagenet')
parser.add_argument('--data_dir', default='./data',
                    help='data dir')

parser.add_argument('--model_save_path', help='Path for saving the trained model')

parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU instead of TPU')

parser.add_argument('--save_freq', type=int, default=10, help='saves the model at end of this many epochs')


# Training details

parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

parser.add_argument('--batch-size', type=int, default=32,
                    help='GLOBAL input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--eval_every', type=int, default=10,
                    help='evaluation frequency ')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=3e-3,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--decay-steps', type=int, default=100000,
                    help='number of learning rate decay steps')
parser.add_argument('--wd', type=float, default=0.03,
                    help='weight decay')
parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='label smoothing')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

# model details
parser.add_argument("--img_size", default=224, type=int,
                    help="Resolution size")
parser.add_argument("--use_moe", action='store_true', default=False,
                    help='use ViT-MoE model')
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                             "ViT-L_32", "ViT-H_14", "ViT-XH_14", "R50-ViT-B_16",
                                             "ViT-MoE-B_16", "ViT-MoE-L_16", "ViT-MoE-H_14",
                                             "ViT-MoE-XH_14"],
                    default="ViT-B_16",
                    help="Which model to use.")
#MOE options
parser.add_argument("--num_experts", type=int, default=8,
                    help='num of experts for Token Mixture Layers')
parser.add_argument("--num_masked_experts", type=float, default=0.0,
                    help='num of experts masked')
parser.add_argument("--capacity_factor", type=float, default=1.0,
                    help='capacity factor of mixer of experts')
parser.add_argument('--top_k', type=int, default=1,
                    help='top_k experts are selected')
parser.add_argument("--use_aux_loss", action='store_true', default=False,
                    help='do not use balanced loss')
parser.add_argument("--aux_loss_alpha", type=float, default=1.0,
                    help='do not use balanced loss')
parser.add_argument("--aux_loss_beta", type=float, default=0.001,
                    help='do not use balanced loss')

parser.add_argument('--switch_deepth', type=int, default=1,
                    help='number of layers used when one time switch done')


parser.add_argument("--mixup", type=float, default=0.5,
                    help='MixUp Augumentation parameter')
parser.add_argument("--beta2", type=float, default=0.999,
                    help='beta2 value of optimizer')
parser.add_argument("--opt", choices=["LAMB", "Adam"],
                    default="Adam",
                    help='Optimizer')
parser.add_argument("--inception_style", action='store_true', default=False,
                    help='use Inception-style preprocessing')
parser.add_argument('--hold_on_epochs', type=float, default=1,
                    help='learning rate (default: 0.01)')

parser.add_argument("--use_representation", action='store_true', default=False,
                    help='use use_representation before head')

parser.add_argument("--share_att", action='store_true', default=False,
                    help='share attention weights')

parser.add_argument("--share_ffn", action='store_true', default=False,
                    help='share attention weights')

parser.add_argument('--group_deepth', type=int, default=128,
                    help='number of layers used within one group')


class MixupAccuracy(nn.Module):
    def forward(self, logits, targets):
        targets = targets['targets_a']
        preds = torch.argmax(logits, dim=-1)
        correct = torch.sum(targets == preds)
        return correct

def build_dali_train(args):
    root = args.data_dir
    train_pat = os.path.join(root, 'train/*')
    train_idx_pat = os.path.join(root, 'idx_files/train/*')
    return DaliDataloader(
        sorted(glob.glob(train_pat)),
        sorted(glob.glob(train_idx_pat)),
        batch_size=args.batch_size,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        gpu_aug=gpc.config.dali.gpu_aug,
        cuda=True,
        mixup_alpha=gpc.config.dali.mixup_alpha,
        randaug_num_layers=2
    )


def build_dali_test(args):
    root = args.data_dir
    val_pat = os.path.join(root, 'validation/*')
    val_idx_pat = os.path.join(root, 'idx_files/validation/*')
    return DaliDataloader(
        sorted(glob.glob(val_pat)),
        sorted(glob.glob(val_idx_pat)),
        batch_size=args.batch_size,
        shard_id=gpc.get_local_rank(ParallelMode.DATA),
        num_shards=gpc.get_world_size(ParallelMode.DATA),
        training=False,
        # gpu_aug=gpc.config.dali.gpu_aug,
        gpu_aug=False,
        cuda=True,
        mixup_alpha=gpc.config.dali.mixup_alpha
    )

def main():
    # Initialize settings
    args = parser.parse_args()

    # Colossal-AI launch from torch
    colossalai.launch_from_torch(config='./config.py')

    # Get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # Build model
    model = Widenet(
        num_experts=args.num_experts,
        capacity_factor=args.capacity_factor
    )

    # Build dataloader
    train_dataloader = build_dali_train(args)
    test_dataloader = build_dali_test(args)

    # Build optimizer
    optimizer = colossalai.nn.Lamb(model.parameters(), lr=args.base_lr, weight_decay=args.wd)

    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    # Learning rate schedule
    lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=5, total_steps=args.epochs)

    # Build trainer
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model, optimizer, criterion, train_dataloader, test_dataloader
    )
    logger.info("initialized colossalai components", ranks=[0])
    trainer = Trainer(engine=engine, logger=logger)

    # Build hooks
    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(accuracy_func=MixupAccuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
        TotalBatchsizeHook(),

        # comment if you do not need to use the hooks below
        #hooks.SaveCheckpointHook(interval=1, checkpoint_dir='args.checkpoint_dir'),
        hooks.TensorboardHook(log_dir='args.log_dir', ranks=[0]),
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        epochs=args.epochs,
        hooks=hook_list,
        display_progress=True,
        test_interval=1
    )


if __name__ == '__main__':
    main()