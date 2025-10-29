# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Sequence, Union

from mmdet.registry import DATASETS
from mmdet.datasets.api_wrappers import COCOPanoptic
from mmdet.datasets.coco import CocoDataset
import re
from mmengine.fileio import get_local_path

import copy


@DATASETS.register_module()
class CocoPanopticPathDataset(CocoDataset):
    classes = ('Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial', 'Gland', 'Malignant Lesion', 'Proximal Tubular', 'Distal Tubular', 'Intercellular')
    thing_classes = ('Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial', 'Gland', 'Malignant Lesion', 'Proximal Tubular', 'Distal Tubular')
    stuff_classes =  ('Intercellular')
    METAINFO = dict(
        classes=classes,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
        palette= [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252)],
        datasets=("DigestPath", "OmniSeg", "GlaS", "PanNuke"),
        dataset2scale = {"DigestPath":0, "OmniSeg":1, "GlaS":2, "PanNuke":3}
    )
    COCOAPI = COCOPanoptic
    # ann_id is not unique in coco panoptic dataset.
    ANN_ID_UNIQUE = False

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=None, ann=None, seg=None),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 backend_args: dict = None,
                 **kwargs) -> None:
        # import pdb; pdb.set_trace()
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            backend_args=backend_args,
            **kwargs)
        

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        self.dataset2label = {dataset: i for i, dataset in enumerate(self.metainfo['datasets'])}
        self.dataset_num = {dataset: 0 for dataset in self.metainfo['datasets']}
        
        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
    
    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``.

        Returns:
            dict: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ret = re.match('(.*?)_.*', img_info['file_name'])
        
        ann_info = raw_data_info['raw_ann_info']
        # filter out unmatched annotations which have
        # same segment_id but belong to other image
        ann_info = [
            ann for ann in ann_info if ann['image_id'] == img_info['img_id']
        ]
        data_info = {}

        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].replace('.jpg', '.png'))
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']
        
        
        data_info['dataset_id'] = self.dataset2label[ret.group(1)]
        data_info['dataset_name'] = ret.group(1)
        data_info['dataset_scale'] = self.metainfo['dataset2scale'][ret.group(1)]
        self.dataset_num[data_info['dataset_name']] += 1

        if self.return_classes:
            data_info['classes'] = list(self.metainfo['classes']) + ['background']
            data_info['text'] = self.metainfo['thing_classes']
            data_info['stuff_text'] = self.metainfo['stuff_classes']
            data_info['custom_entities'] = True  # no important
        # import pdb; pdb.set_trace()
        instances = []
        segments_info = []
        for ann in ann_info:
            instance = {}
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                instance['bbox'] = bbox
                instance['bbox_label'] = contiguous_cat_id
                if not is_crowd:
                    instance['ignore_flag'] = 0
                else:
                    instance['ignore_flag'] = 1
                    is_thing = False

            segment_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            segments_info.append(segment_info)
            if len(instance) > 0 and is_thing:
                instances.append(instance)
        data_info['instances'] = instances
        data_info['segments_info'] = segments_info
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter images too small or without ground truth.

        Returns:
            List[dict]: ``self.data_list`` after filtering.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        ids_with_ann = set()
        # check whether images have legal thing annotations.
        for data_info in self.data_list:
            for segment_info in data_info['segments_info']:
                if not segment_info['is_thing']:
                    continue
                ids_with_ann.add(data_info['img_id'])

        valid_data_list = []
        for data_info in self.data_list:
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(width, height) >= min_size:
                valid_data_list.append(data_info)

        return valid_data_list
