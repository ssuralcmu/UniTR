import pickle
import time

import numpy as np
import torch
import tqdm
from PIL import Image
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_map_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    map_output_dir = None
    if getattr(args, "save_map_outputs", False):
        map_output_dir = _prepare_map_output_dirs(result_dir)

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    all_preds_dict = []
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            preds_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
        all_preds_dict.extend(preds_dict)

        if map_output_dir is not None:
            _save_map_segmentation_batch_outputs(
                output_dir=map_output_dir,
                class_names=dataset.map_classes,
                preds_dict=preds_dict,
                metadata=batch_dict.get('metadata', None),
                score_thresh=float(getattr(args, "map_score_thresh", 0.5)),
            )
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        all_preds_dict = common_utils.merge_results_dist(all_preds_dict, len(dataset), tmpdir=result_dir / 'tmpdir')
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    if map_output_dir is not None:
        logger.info(f"Map segmentation outputs are saved to {map_output_dir}")


    metric = dataset.evaluation_map_segmentation(
        all_preds_dict
    )
    print(metric)
    logger.info('****************Evaluation done.*****************')
    return metric

def _build_bevfusion_map_palette(class_names):
    """RGB palette for the exact 6 relevant BEVFusion nuScenes map classes."""
    bevfusion_six_class_palette = {
        "drivable_area": (166, 206, 227),
        "ped_crossing": (251, 154, 153),
        "walkway": (227, 26, 28),
        "stop_line": (253, 191, 111),
        "carpark_area": (255, 127, 0),
        "divider": (106, 61, 154),
    }

    missing_classes = [name for name in class_names if name not in bevfusion_six_class_palette]
    if len(missing_classes) > 0:
        raise KeyError(
            "Only these classes are supported for export palette: "
            f"{list(bevfusion_six_class_palette.keys())}. "
            f"Got extra classes: {missing_classes}"
        )

    return [bevfusion_six_class_palette[name] for name in class_names]




def _save_map_segmentation_outputs(result_dir, dataset, all_preds_dict, score_thresh=0.5):
    """Backward-compatible map export entrypoint (for older call sites)."""
    output_dir = _prepare_map_output_dirs(result_dir)
    class_names = getattr(dataset, "map_classes", None)
    if class_names is None:
        raise AttributeError("Dataset has no map_classes, cannot export map outputs")

    # Fallback sequential names when metadata is unavailable in this legacy path.
    fallback_metadata = [{"token": f"sample_{idx:06d}"} for idx in range(len(all_preds_dict))]
    _save_map_segmentation_batch_outputs(
        output_dir=output_dir,
        class_names=class_names,
        preds_dict=all_preds_dict,
        metadata=fallback_metadata,
        score_thresh=score_thresh,
    )
    return output_dir

def _prepare_map_output_dirs(result_dir):
    output_dir = result_dir / "map_outputs"
    probs_dir = output_dir / "probs"
    masks_dir = output_dir / "masks"
    vis_dir = output_dir / "vis_bevfusion"
    for cur_dir in [probs_dir, masks_dir, vis_dir]:
        cur_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def _parse_tokens_from_metadata(metadata, num_preds):
    if metadata is None:
        return [f"sample_{idx:06d}" for idx in range(num_preds)]

    if isinstance(metadata, np.ndarray):
        metadata = metadata.tolist()

    tokens = []
    for idx in range(num_preds):
        item = metadata[idx]
        token = item.get('token', None) if isinstance(item, dict) else None
        if token is None:
            token = f"sample_{idx:06d}"
        tokens.append(token)

    return tokens


def _save_map_segmentation_batch_outputs(output_dir, class_names, preds_dict, metadata, score_thresh=0.5):
    probs_dir = output_dir / "probs"
    masks_dir = output_dir / "masks"
    vis_dir = output_dir / "vis_bevfusion"

    palette = _build_bevfusion_map_palette(class_names)
    tokens = _parse_tokens_from_metadata(metadata, len(preds_dict))

    for idx, pred in enumerate(preds_dict):
        token = tokens[idx]
        probs = pred["masks_bev"].detach().cpu().float().numpy()
        masks = probs >= score_thresh

        np.save(probs_dir / f"{token}.npy", probs.astype(np.float32))
        np.save(masks_dir / f"{token}.npy", masks.astype(np.bool_))

        color_img = np.full((masks.shape[1], masks.shape[2], 3), 255, dtype=np.uint8)
        for class_idx, color in enumerate(palette):
            color_img[masks[class_idx]] = np.array(color, dtype=np.uint8)

        vis_img = Image.fromarray(color_img)
        if vis_img.size != (200, 200):
            vis_img = vis_img.resize((200, 200), resample=Image.NEAREST)
        vis_img.save(vis_dir / f"{token}.png")

if __name__ == '__main__':
    pass
