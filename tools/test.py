import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path
import copy

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank', type=int, default=0)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')
    parser.add_argument('--eval_map',  action='store_true', default=False, help='evaluate bev map segmentation')
    parser.add_argument('--save_map_outputs', action='store_true', default=False,
                        help='save map segmentation logits, binary masks and colorized masks')
    parser.add_argument('--map_score_thresh', type=float, default=0.5,
                        help='threshold to binarize per-class map probabilities')

    parser.add_argument('--eval_context_subsets', type=str, default='all',
                        help='comma-separated context subsets for NuScenes eval: night_rain,all,night,rain,clear')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    
    # start evaluation
    if args.eval_map:
        eval_utils.eval_map_one_epoch(
            cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
            result_dir=eval_output_dir
        )
    else:
        eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
            result_dir=eval_output_dir
        )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        if args.eval_map:
             tb_dict = eval_utils.eval_map_one_epoch(
                cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
                result_dir=cur_result_dir
            )
        else:
            tb_dict = eval_utils.eval_one_epoch(
                cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
                result_dir=cur_result_dir
            )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)

def parse_eval_context_subsets(eval_context_subsets):
    if eval_context_subsets is None:
        return None

    alias = {
        'all': 'all',
        'night': 'night',
        'rain': 'rain',
        'night_rain': 'night_rain',
        'rain_night': 'night_rain',
        'rain+night': 'night_rain',
        'night+rain': 'night_rain',
        'clear': 'clear',
        'no_rain_no_night': 'clear',
        'no_night_no_rain': 'clear'
    }

    subsets = []
    for raw_name in eval_context_subsets.split(','):
        name = raw_name.strip().lower().replace('-', '_').replace(' ', '_')
        if not name:
            continue
        name = alias.get(name, name)
        if name not in ['all', 'night', 'rain', 'night_rain', 'clear']:
            raise ValueError(f'Unsupported context subset: {raw_name}')
        if name not in subsets:
            subsets.append(name)

    return subsets or None


def build_eval_output_dir(output_dir, args):
    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    return eval_output_dir

def main():
    args, cfg = parse_config()

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = build_eval_output_dir(output_dir, args)

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    context_subsets = parse_eval_context_subsets(args.eval_context_subsets)
    if context_subsets is None:
        context_subsets = [None]

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    for context_subset in context_subsets:
        cur_eval_output_dir = eval_output_dir
        if context_subset is not None:
            if not hasattr(cfg.DATA_CONFIG, 'CONTEXT_CONFIG') or cfg.DATA_CONFIG.CONTEXT_CONFIG is None:
                raise ValueError('eval_context_subsets requires DATA_CONFIG.CONTEXT_CONFIG in cfg')
            cfg.DATA_CONFIG.CONTEXT_CONFIG.EVAL_SUBSET = context_subset
            cur_eval_output_dir = eval_output_dir / f'context_subset_{context_subset}'

        cur_eval_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info('Start evaluation with context subset: %s', context_subset if context_subset is not None else 'default')
        log_config_to_file(cfg, logger=logger)

        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )

        model_cfg = copy.deepcopy(cfg.MODEL)
        model = build_network(model_cfg=model_cfg, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        with torch.no_grad():
            if args.eval_all:
                repeat_eval_ckpt(model, test_loader, args, cur_eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
            else:
                num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
                epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
                eval_single_ckpt(model, test_loader, args, cur_eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
