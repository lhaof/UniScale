# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable
import torch
from mmdet.datasets.api_wrappers import COCOPanoptic
from mmdet.registry import METRICS
import multiprocessing
from scipy.optimize import linear_sum_assignment
from PIL import Image
import os
import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
from tabulate import tabulate
import scipy.io as sio
import cv2
INSTANCE_OFFSET = 1000
try:
    import panopticapi
    from panopticapi.evaluation import VOID, PQStat
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    panopticapi = None
    id2rgb = None
    rgb2id = None
    VOID = None
    PQStat = None

class NucleiState():
    def __init__(self):
        self.PQ = 0
        self.DICE = 0
        self.AJI = 0
        self.n_sample = 0

    def __iadd__(self, nuclei_stat):
        self.PQ += nuclei_stat.PQ
        self.DICE += nuclei_stat.DICE
        self.AJI += nuclei_stat.AJI
        self.n_sample += nuclei_stat.n_sample
        return self

    @property
    def PQ_average(self):
        return self.PQ / self.n_sample if self.n_sample != 0 else 0

    @property
    def DICE_average(self):
        return self.DICE / self.n_sample if self.n_sample != 0 else 0

    @property
    def AJI_average(self):
        return self.AJI / self.n_sample if self.n_sample != 0 else 0


class SemState():
    def __init__(self):
        self.area_intersect = 0
        self.area_union = 0
        self.area_pred_label = 0
        self.area_label = 0
        self.mDice = 0
        self.mIoU = 0
        self.MPA = 0
        self.n_sample = 0

    def __iadd__(self, sem_stat):
        self.area_intersect += sem_stat.area_intersect
        self.area_union += sem_stat.area_union
        self.area_pred_label += sem_stat.area_pred_label
        self.area_label += sem_stat.area_label
        self.mDice += sem_stat.mDice
        self.mIoU += sem_stat.mIoU
        self.MPA += sem_stat.MPA
        self.n_sample += sem_stat.n_sample
        return self

    @property
    def mIoU_average(self):
        return self.mIoU / self.n_sample if self.n_sample != 0 else 0
    
    @property
    def mDice_average(self):
        return self.mDice / self.n_sample if self.n_sample != 0 else 0

    @property
    def MPA_average(self):
        return self.MPA / self.n_sample if self.n_sample != 0 else 0
    
    @property
    def IoU(self):
        return self.area_intersect / self.area_union if self.area_union != 0 else 0
    
    @property
    def Dice(self):
        return 2 * self.area_intersect / (self.area_pred_label + self.area_label) if self.area_pred_label + self.area_label != 0 else 0
    
    @property
    def Precision(self):
        return self.area_intersect / self.area_pred_label if self.area_pred_label != 0 else 0
    
    @property
    def Recall(self):
        return self.area_intersect / self.area_label if self.area_label != 0 else 0
    
    @property
    def F1(self):
        precision = self.Precision
        recall = self.Recall
        beta = 1
        try:
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
        except:
            score = 0
        return score
    
    @property
    def Accuracy(self):
        return self.area_intersect / self.area_label if self.area_label != 0 else 0


@METRICS.register_module()
class NucleiSegMetric(BaseMetric):
    """COCO panoptic segmentation evaluation metric.

    Evaluate PQ, SQ RQ for panoptic segmentation tasks. Please refer to
    https://cocodataset.org/#panoptic-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        seg_prefix (str, optional): Path to the directory which contains the
            coco panoptic segmentation mask. It should be specified when
            evaluate. Defaults to None.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created.
            It should be specified when format_only is True. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        nproc (int): Number of processes for panoptic quality computing.
            Defaults to 32. When ``nproc`` exceeds the number of cpu cores,
            the number of cpu cores is used.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'coco_panoptic'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 seg_prefix: Optional[str] = None,
                 classwise: bool = False,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 nproc: int = 32,
                 Comp_Sem: bool = False,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        if panopticapi is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super().__init__(collect_device=collect_device, prefix=prefix)
        self.classwise = classwise
        self.format_only = format_only
        self.Comp_Sem = Comp_Sem
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.tmp_dir = None
        # outfile_prefix should be a prefix of a path which points to a shared
        # storage when train or test with multi nodes.
        self.outfile_prefix = outfile_prefix
        if outfile_prefix is None:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.outfile_prefix = osp.join(self.tmp_dir.name, 'results')
        # the directory to save predicted panoptic segmentation mask
        self.seg_out_dir = f'{self.outfile_prefix}.panoptic'

        self.nproc = nproc
        self.seg_prefix = seg_prefix

        self.cat_ids = None
        self.cat2label = None

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        if ann_file:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCOPanoptic(local_path)
            self.categories = self._coco_api.cats
        else:
            self._coco_api = None
            self.categories = None

    def __del__(self) -> None:
        """Clean up."""
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> Tuple[str, str]:
        """Convert ground truth to coco panoptic segmentation format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset.
            outfile_prefix (str): The filename prefix of the json file. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".

        Returns:
            Tuple[str, str]: The filename of the json file and the name of the\
                directory which contains panoptic segmentation masks.
        """
        assert len(gt_dicts) > 0, 'gt_dicts is empty.'
        gt_folder = osp.dirname(gt_dicts[0]['seg_map_path'])
        converted_json_path = f'{outfile_prefix}.gt.json'

        categories = []
        for id, name in enumerate(self.dataset_meta['classes']):
            isthing = 1 if name in self.dataset_meta['thing_classes'] else 0
            categories.append({'id': id, 'name': name, 'isthing': isthing})

        image_infos = []
        annotations = []
        for gt_dict in gt_dicts:
            img_id = gt_dict['image_id']
            image_info = {
                'id': img_id,
                'width': gt_dict['width'],
                'height': gt_dict['height'],
                'file_name': osp.split(gt_dict['seg_map_path'])[-1]
            }
            image_infos.append(image_info)

            pan_png = mmcv.imread(gt_dict['seg_map_path']).squeeze()
            pan_png = pan_png[:, :, ::-1]
            pan_png = rgb2id(pan_png)
            segments_info = []
            for segment_info in gt_dict['segments_info']:
                id = segment_info['id']
                label = segment_info['category']
                mask = pan_png == id
                isthing = categories[label]['isthing']
                if isthing:
                    iscrowd = 1 if not segment_info['is_thing'] else 0
                else:
                    iscrowd = 0

                new_segment_info = {
                    'id': id,
                    'category_id': label,
                    'isthing': isthing,
                    'iscrowd': iscrowd,
                    'area': mask.sum()
                }
                segments_info.append(new_segment_info)

            segm_file = image_info['file_name'].replace('.jpg', '.png')
            annotation = dict(
                image_id=img_id,
                segments_info=segments_info,
                file_name=segm_file)
            annotations.append(annotation)
            pan_png = id2rgb(pan_png)

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmdet CocoPanopticMetric.'
        )
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
        )
        if len(annotations) > 0:
            coco_json['annotations'] = annotations
        dump(coco_json, converted_json_path)
        return converted_json_path, gt_folder

    def result2json(self, results: Sequence[dict],
                    outfile_prefix: str) -> Tuple[str, str]:
        """Dump the panoptic results to a COCO style json file and a directory.

        Args:
            results (Sequence[dict]): Testing results of the dataset.
            outfile_prefix (str): The filename prefix of the json files and the
                directory.

        Returns:
            Tuple[str, str]: The json file and the directory which contains \
                panoptic segmentation masks. The filename of the json is
                "somepath/xxx.panoptic.json" and name of the directory is
                "somepath/xxx.panoptic".
        """
        label2cat = dict((v, k) for (k, v) in self.cat2label.items())
        pred_annotations = []
        for idx in range(len(results)):
            result = results[idx]
            for segment_info in result['segments_info']:
                sem_label = segment_info['category_id']
                # convert sem_label to json label
                cat_id = label2cat[sem_label]
                segment_info['category_id'] = label2cat[sem_label]
                is_thing = self.categories[cat_id]['isthing']
                segment_info['isthing'] = is_thing
            pred_annotations.append(result)
        pan_json_results = dict(annotations=pred_annotations)
        # import pdb; pdb.set_trace()
        json_filename = f'{outfile_prefix}_panoptic.json'
        dump(pan_json_results, json_filename)
        return json_filename, (
            self.seg_out_dir
            if self.tmp_dir is None else tempfile.gettempdir())

    def _parse_predictions(self,
                           pred: dict,
                           img_id: int,
                           segm_file: str,
                           label2cat=None) -> dict:
        """Parse panoptic segmentation predictions.

        Args:
            pred (dict): Panoptic segmentation predictions.
            img_id (int): Image id.
            segm_file (str): Segmentation file name.
            label2cat (dict): Mapping from label to category id.
                Defaults to None.

        Returns:
            dict: Parsed predictions.
        """
        # import pdb; pdb.set_trace()
        result = dict()
        result['img_id'] = img_id
        # shape (1, H, W) -> (H, W)
        pan = pred['pred_panoptic_seg']['sem_seg'].cpu().numpy()[0]
        ignore_index = pred['pred_panoptic_seg'].get(
            'ignore_index', len(self.dataset_meta['classes']))
        pan_labels = np.unique(pan)
        segments_info = []
        for pan_label in pan_labels:
            sem_label = pan_label % INSTANCE_OFFSET
            # We reserve the length of dataset_meta['classes']
            # and ignore_index for VOID label
            if sem_label == len(
                    self.dataset_meta['classes']) or sem_label == ignore_index:
                continue
            mask = pan == pan_label
            area = mask.sum()
            segments_info.append({
                'id':
                int(pan_label),
                # when ann_file provided, sem_label should be cat_id, otherwise
                # sem_label should be a continuous id, not the cat_id
                # defined in dataset
                'category_id':
                label2cat[sem_label] if label2cat else sem_label,
                'area':
                int(area)
            })
        # evaluation script uses 0 for VOID label.
        pan[pan % INSTANCE_OFFSET == len(self.dataset_meta['classes'])] = VOID
        pan[pan % INSTANCE_OFFSET == ignore_index] = VOID

        pan = id2rgb(pan).astype(np.uint8)
        mmcv.imwrite(pan[:, :, ::-1], osp.join(self.seg_out_dir, segm_file))
        result = {
            'image_id': img_id,
            'segments_info': segments_info,
            'file_name': segm_file
        }

        return result

    def _compute_batch_pq_stats(self, data_samples: Sequence[dict]):
        """Process gts and predictions when ``outfile_prefix`` is not set, gts
        are from dataset or a json file which is defined by ``ann_file``.

        Intermediate results, ``pq_stats``, are computed here and put into
        ``self.results``.
        """
        if self._coco_api is None:
            categories = dict()
            for id, name in enumerate(self.dataset_meta['classes']):
                isthing = 1 if name in self.dataset_meta['thing_classes']\
                    else 0
                categories[id] = {'id': id, 'name': name, 'isthing': isthing}
            label2cat = None
        else:
            categories = self.categories
            cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
            label2cat = {i: cat_id for i, cat_id in enumerate(cat_ids)}

        for data_sample in data_samples:
            # parse pred
            img_id = data_sample['img_id']
            segm_file = osp.basename(data_sample['img_path']).replace(
                '.jpg', '.png')
            result = self._parse_predictions(
                pred=data_sample,
                img_id=img_id,
                segm_file=segm_file,
                label2cat=label2cat)

            # parse gt
            gt = dict()
            gt['image_id'] = img_id
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['file_name'] = segm_file

            if self._coco_api is None:
                # get segments_info from data_sample
                seg_map_path = osp.join(self.seg_prefix, segm_file)
                pan_png = mmcv.imread(seg_map_path).squeeze()
                pan_png = pan_png[:, :, ::-1]
                pan_png = rgb2id(pan_png)
                segments_info = []

                for segment_info in data_sample['segments_info']:
                    id = segment_info['id']
                    label = segment_info['category']
                    mask = pan_png == id
                    isthing = categories[label]['isthing']
                    if isthing:
                        iscrowd = 1 if not segment_info['is_thing'] else 0
                    else:
                        iscrowd = 0

                    new_segment_info = {
                        'id': id,
                        'category_id': label,
                        'isthing': isthing,
                        'iscrowd': iscrowd,
                        'area': mask.sum()
                    }
                    segments_info.append(new_segment_info)
            else:
                # get segments_info from annotation file
                segments_info = self._coco_api.imgToAnns[img_id]

            gt['segments_info'] = segments_info
            ignore_categories = {el['id']: el for el in categories.values() if el['isthing'] == 0}
            
            pq_stats = nuclei_metric_compute_single_core(
                proc_id=0,
                annotation_set=[(gt, result)],
                gt_folder=self.seg_prefix,
                pred_folder=self.seg_out_dir,
                ignore_index=ignore_categories)


            self.results.append(pq_stats)

    def _process_gt_and_predictions(self, data_samples: Sequence[dict]):
        """Process gts and predictions when ``outfile_prefix`` is set.

        The predictions will be saved to directory specified by
        ``outfile_predfix``. The matched pair (gt, result) will be put into
        ``self.results``.
        """
        for data_sample in data_samples:
            # parse pred
            img_id = data_sample['img_id']
            segm_file = osp.basename(data_sample['img_path']).replace(
                '.jpg', '.png')
            result = self._parse_predictions(
                pred=data_sample, img_id=img_id, segm_file=segm_file)

            # parse gt
            gt = dict()
            gt['image_id'] = img_id
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]

            if self._coco_api is None:
                # get segments_info from dataset
                gt['segments_info'] = data_sample['segments_info']
                gt['seg_map_path'] = data_sample['seg_map_path']

            self.results.append((gt, result))

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        # If ``self.tmp_dir`` is none, it will save gt and predictions to
        # self.results, otherwise, it will compute pq_stats here.
        # import pdb; pdb.set_trace()
        if self.tmp_dir is None:
            self._process_gt_and_predictions(data_samples)
        else:
            self._process_gt_and_predictions(data_samples)
            # self._compute_batch_pq_stats(data_samples)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch. There
                are two cases:

                - When ``outfile_prefix`` is not provided, the elements in
                  results are pq_stats which can be summed directly to get PQ.
                - When ``outfile_prefix`` is provided, the elements in
                  results are tuples like (gt, pred).

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        # import pdb; pdb.set_trace() 
        if True:
            # do evaluation after collect all the results
            # import pdb; pdb.set_trace()  
            # split gt and prediction list
            gts, preds = zip(*results)

            if self._coco_api is None:
                # use converted gt json file to initialize coco api
                logger.info('Converting ground truth to coco format...')
                coco_json_path, gt_folder = self.gt_to_coco_json(
                    gt_dicts=gts, outfile_prefix=self.outfile_prefix)
                self._coco_api = COCOPanoptic(coco_json_path)
            else:
                gt_folder = self.seg_prefix

            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
            self.cat2label = {
                cat_id: i
                for i, cat_id in enumerate(self.cat_ids)
            }
            self.img_ids = self._coco_api.get_img_ids()
            self.categories = self._coco_api.cats

            # convert predictions to coco format and dump to json file
            json_filename, pred_folder = self.result2json(
                results=preds, outfile_prefix=self.outfile_prefix)

            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(self.outfile_prefix)}')
                return dict()

            imgs = self._coco_api.imgs
            gt_json = self._coco_api.img_ann_map
            gt_json = [{
                'image_id': k,
                'segments_info': v,
                'file_name': imgs[k]['segm_file']
            } for k, v in gt_json.items()]
            try:
                pred_json = load(json_filename)
                pred_json = dict(
                    (el['image_id'], el) for el in pred_json['annotations'])
            except:
                res = {}
                res["PQ"] = 0
                res["AJI"] = 0
                res["DICE"] = 0
                results = OrderedDict({"nuclei_seg": res})

                _print_nuclei_metrics(res, logger)

                return res

            # match the gt_anns and pred_anns in the same image
            matched_annotations_list = []
            for gt_ann in gt_json:
                img_id = gt_ann['image_id']
                if img_id not in pred_json.keys():
                    raise Exception('no prediction for the image'
                                    ' with id: {}'.format(img_id))
                matched_annotations_list.append((gt_ann, pred_json[img_id]))
            # import pdb; pdb.set_trace()
            ignore_categories = {el['id']: el for el in self.categories.values() if el['isthing'] == 0}
            nuclei_res = nuclei_metric_compute_single_core(
                0,
                matched_annotations_list,
                gt_folder,
                pred_folder,
                ignore_categories,
                Comp_Sem=self.Comp_Sem
                )
            # nuclei_res = nuclei_metric_compute_multi_core(
            #     matched_annotations_list,
            #     gt_folder,
            #     pred_folder,
            #     ignore_categories,
            #     Comp_Sem=self.Comp_Sem,
            #     seg_out_dir=self.seg_out_dir)
            if self.Comp_Sem:
                nuclei_res, sem_res = nuclei_res


        res = {}
        res["PQ"] = 100 * nuclei_res.PQ_average
        res["AJI"] = 100 * nuclei_res.AJI_average
        res["DICE"] = 100 * nuclei_res.DICE_average
        results = OrderedDict({"nuclei_seg": res})

        _print_nuclei_metrics(res, logger)
        if self.Comp_Sem:
            _print_sem_metrics(sem_res, logger)
        return res
    
def _print_sem_metrics(sem_stat, logger):
    headers = ["", "IoU", "mIoU","Dice","mDice", "Precision", "Recall", "F1", "Accuracy", "MPA"]
    data = [["Semantic Segmentation"] + [sem_stat.IoU] + [sem_stat.mIoU_average] + [sem_stat.Dice] + [sem_stat.mDice_average] + [sem_stat.Precision] + [sem_stat.Recall] + [sem_stat.F1] + [sem_stat.Accuracy] + [sem_stat.MPA_average]]
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".5f", stralign="center", numalign="center"
    )
    print_log("Semantic segmentation metrics:\n" + table, logger=logger, level=logging.INFO)

def _print_nuclei_metrics(nuclei_stat, logger):
    headers = ["", "PQ", "AJI", "DICE"]
    data = [["Nuclei Segmentation"] + [nuclei_stat["PQ"]] + [nuclei_stat["AJI"]] + [nuclei_stat["DICE"]]]
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    print_log("Nuclei segmentation metrics:\n" + table, logger=logger, level=logging.INFO)

def parse_pq_results(pq_results: dict) -> dict:
    """Parse the Panoptic Quality results.

    Args:
        pq_results (dict): Panoptic Quality results.

    Returns:
        dict: Panoptic Quality results parsed.
    """
    result = dict()
    result['PQ'] = 100 * pq_results['All']['pq']
    result['SQ'] = 100 * pq_results['All']['sq']
    result['RQ'] = 100 * pq_results['All']['rq']
    result['PQ_th'] = 100 * pq_results['Things']['pq']
    result['SQ_th'] = 100 * pq_results['Things']['sq']
    result['RQ_th'] = 100 * pq_results['Things']['rq']
    result['PQ_st'] = 100 * pq_results['Stuff']['pq']
    result['SQ_st'] = 100 * pq_results['Stuff']['sq']
    result['RQ_st'] = 100 * pq_results['Stuff']['rq']
    return result


def print_panoptic_table(
        pq_results: dict,
        classwise_results: Optional[dict] = None,
        logger: Optional[Union['MMLogger', str]] = None) -> None:
    """Print the panoptic evaluation results table.

    Args:
        pq_results(dict): The Panoptic Quality results.
        classwise_results(dict, optional): The classwise Panoptic Quality.
            results. The keys are class names and the values are metrics.
            Defaults to None.
        logger (:obj:`MMLogger` | str, optional): Logger used for printing
            related information during evaluation. Default: None.
    """

    headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
    data = [headers]
    for name in ['All', 'Things', 'Stuff']:
        numbers = [
            f'{(pq_results[name][k] * 100):0.3f}' for k in ['pq', 'sq', 'rq']
        ]
        row = [name] + numbers + [pq_results[name]['n']]
        data.append(row)
    table = AsciiTable(data)
    print_log('Panoptic Evaluation Results:\n' + table.table, logger=logger)

    if classwise_results is not None:
        class_metrics = [(name, ) + tuple(f'{(metrics[k] * 100):0.3f}'
                                          for k in ['pq', 'sq', 'rq'])
                         for name, metrics in classwise_results.items()]
        num_columns = min(8, len(class_metrics) * 4)
        results_flatten = list(itertools.chain(*class_metrics))
        headers = ['category', 'PQ', 'SQ', 'RQ'] * (num_columns // 4)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        data = [headers]
        data += [result for result in results_2d]
        table = AsciiTable(data)
        print_log(
            'Classwise Panoptic Evaluation Results:\n' + table.table,
            logger=logger)



def nuclei_metric_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, Comp_Sem=False, seg_out_dir=None):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    # print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(nuclei_metric_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories, Comp_Sem, seg_out_dir))
        processes.append(p)
    nuclei_stat = NucleiState()
    if Comp_Sem:
        sem_stat = SemState()
        for p in processes:
            nuclei_stat += p.get()[0]
            sem_stat += p.get()[1]
    else:
        for p in processes:
            nuclei_stat += p.get()
    return nuclei_stat, sem_stat if Comp_Sem else nuclei_stat

def nuclei_metric_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, ignore_index, Comp_Sem=False, seg_out_dir=None):
    nuclei_stat = NucleiState()
    if Comp_Sem:
        sem_stat = SemState()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        # import pdb; pdb.set_trace()
        # if idx % 100 == 0:
            # print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        try:
            pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        except:
            print("error result")
            pan_pred = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_pred = rgb2id(pan_pred)

        for el in gt_ann['segments_info']:
            if el['category_id'] in ignore_index:
                pan_gt[pan_gt == el['id']] = 0
        for el in pred_ann['segments_info']:
            if el['category_id'] in ignore_index:
                pan_pred[pan_pred == el['id']] = 0
                
        # sio.savemat(os.path.join(seg_out_dir, "GT", gt_ann['file_name'].replace(".png", ".mat")), {"inst_map": pan_gt})
        


        type_map = pan_pred.copy()
        inst_centroid = []
        inst_type = []
        # import pdb; pdb.set_trace()
        for el in pred_ann['segments_info']:
            if el['category_id'] in ignore_index:
                continue
            type_map[pan_pred == el['id']] = el['category_id']

            inst_map = np.zeros_like(pan_pred).astype(np.uint8)
            inst_map[pan_pred == el['id']] = 1
            contours, _ = cv2.findContours(inst_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    inst_centroid.append((cX, cY))
            
            inst_type.append(el['category_id'])
        
        inst_centroid = np.array(inst_centroid)
        inst_type = np.array(inst_type)

        pan_gt = remap_label(pan_gt)
        pan_pred = remap_label(pan_pred)

        try:
            pq_info = get_fast_pq(pan_gt, pan_pred)[0]
            nuclei_stat.PQ += pq_info[2]
            nuclei_stat.AJI += get_fast_aji(pan_gt, pan_pred)
            nuclei_stat.DICE += get_dice_1(pan_gt, pan_pred)
            nuclei_stat.n_sample += 1
            
            if Comp_Sem:
                num_classes = 2
                pred_label = pan_pred
                label = pan_gt
                pred_label[pred_label > 0] = 1
                label[label > 0] = 1
                pred_label = torch.from_numpy(pred_label)
                label = torch.from_numpy(label)
                
                intersect = pred_label[pred_label == label]
                area_intersect = torch.histc(
                    intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
                area_pred_label = torch.histc(
                    pred_label.float(), bins=num_classes, min=0, max=num_classes - 1)
                area_label = torch.histc(
                    label.float(), bins=num_classes, min=0, max=num_classes - 1)
                area_union = area_pred_label + area_label - area_intersect
                
                sem_stat.area_intersect += area_intersect[1]
                sem_stat.area_union += area_union[1]
                sem_stat.area_pred_label += area_pred_label[1]
                sem_stat.area_label += area_label[1]
                dice = 2 * area_intersect[1] / (area_pred_label[1] + area_label[1])
                iou = area_intersect[1] / area_union[1]
                acc = area_intersect[1] / area_pred_label[1]
                sem_stat.mIoU += iou
                sem_stat.mDice += dice
                sem_stat.MPA += acc
                sem_stat.n_sample += 1
        except:
            continue
            # nuclei_stat.PQ += 0
            # nuclei_stat.AJI += 0
            # nuclei_stat.DICE += 0
            # nuclei_stat.n_sample += 1
    return nuclei_stat, sem_stat if Comp_Sem else nuclei_stat



def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * np.sum(inter) / np.sum(denom)


def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.

    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score