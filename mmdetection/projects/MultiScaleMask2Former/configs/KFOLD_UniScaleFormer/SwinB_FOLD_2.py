import os

_base_ = [
    'mmdet::mask2former/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic.py'
]

custom_imports = dict(imports=['projects.MultiScaleMask2Former.evaluation', 'projects.MultiScaleMask2Former.datasets',
                               'projects.MultiScaleMask2Former.models'])

classes = ('Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial', 'Gland', 'Malignant Lesion', 'Proximal Tubular', 'Distal Tubular', 'Nuclei', 'Intercellular')
thing_classes = ('Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial', 'Gland', 'Malignant Lesion', 'Proximal Tubular', 'Distal Tubular', 'Nuclei')
stuff_classes =  ('Intercellular',)

num_things_classes = len(thing_classes)
num_stuff_classes = len(stuff_classes)
num_scale = 4

num_query = 200

num_classes = num_stuff_classes + num_things_classes

multi_head = False

feat_dim = 512

root_p = "/YOUR_PATH/"
fold = 'fold_2'

model = dict(
    panoptic_head=dict(
        type='AuxScaleClipMask2FormerHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        auxscaleloss=True,
        scale_query_cross=False,
        num_queries=num_query,
        clip_encoder=dict(
            dim_lang = 512,
            dim_projection = 512,
            max_token_num = 77,
            lang_encoder = {'NAME': 'ClipTextEncoder', "TOKENIZER": "clip", 'CONTEXT_LENGTH': 77, 
                            'WIDTH': 512, 'HEADS': 8, 'LAYERS': 12, 'AUTOGRESSIVE': True, 'PRETRAINED_TOKENIZER': f"{root_p}/clip-vit-base-patch32"},
            tokenizer_type = "clip",
            verbose = True,
            load_from=f"{root_p}/pretrain_check/lang_encoder.pt",
            freeze=True,
            seem=True
        ),
        multi_head=multi_head,
        feat_channels=feat_dim,
        out_channels=feat_dim,
        pixel_decoder=dict(
            encoder=dict(
                    num_layers=6,
                    layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                        self_attn_cfg=dict(  # MultiScaleDeformableAttention
                            embed_dims=feat_dim,
                        ),
                        ffn_cfg=dict(
                            embed_dims=feat_dim,
                        )
                    )
                ),
            positional_encoding=dict(num_feats=256, normalize=True),
            ),
        positional_encoding=dict(num_feats=256, normalize=True),
        transformer_decoder=dict(
                num_layers=6,
                layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=feat_dim,
                    ),
                    cross_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=feat_dim,
                    ),
                    ffn_cfg=dict(
                        embed_dims=feat_dim,
                    )
                )
            ),
        loss_scale=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=3.0,
            reduction='mean',
            class_weight=[1.0] * num_scale),
        loss_cls=dict(
            class_weight=[1.0]*num_classes + [0.1]
        ),
    ),
    
    panoptic_fusion_head=dict(
        num_stuff_classes=num_stuff_classes,
        num_things_classes=num_things_classes
    ),
    test_cfg=dict(
        max_per_image=num_query,
    )
)



metainfo = dict(
    classes=classes,
    thing_classes=thing_classes,
    stuff_classes=stuff_classes,
    palette= [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
     (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
     (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
     (165, 42, 42), (255, 77, 255), (0, 226, 252)],
    datasets=("DigestPath", "OmniSeg", "GlaS", "PanNuke", "NuInsSeg"),
    dataset2scale = {"DigestPath":0, "OmniSeg":1, "GlaS":2, "PanNuke":3, "NuInsSeg":3}
)

image_size = (1024, 1024)
data_root = f"{root_p}/Folded_Data/Processed_Data_PGDON_{fold}"
dataset_type = 'CocoPanopticPathDataset'
backend_args = None

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args={{_base_.backend_args}}),
    dict(type='RandomFlip', prob=0.5),
    # large scale jittering
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.8, 1.5),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction', 
                            'dataset_id', 'dataset_name', 'classes', 'dataset_scale')) 
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'dataset_id', 'dataset_name', 'classes', 'dataset_scale'))
]


train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler_SameDataset', drop_last=True,
        dataset_weights=[1, 1, 1, 3, 3]),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        ann_file='annotations/panoptic_train.json',
        data_prefix=dict(
            img='train_image/', seg='train_panoptic_image/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args),
    )


_data_root = f"{root_p}/Folded_Data/Processed_Data_PGDON_P_{fold}/"
dataset_PanNuke = dict(
        type=dataset_type,
        data_root=_data_root,
        metainfo=metainfo,
        return_classes=True,
        ann_file='annotations/panoptic_val.json',
        data_prefix=dict(img='val_image/', seg='val_panoptic_image/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
val_evaluator_PanNuke = dict(
        type='NucleiSegMetric',
        ann_file=_data_root + 'annotations/panoptic_val.json',
        seg_prefix=_data_root + 'val_panoptic_image/',
        outfile_prefix=_data_root + 'tmp/results',
        Comp_Sem = True,
        backend_args=backend_args)

_data_root = f"{root_p}/Folded_Data/Processed_Data_PGDON_G_{fold}/"
dataset_GlaS = dict(
        type=dataset_type,
        data_root=_data_root,
        metainfo=metainfo,
        return_classes=True,
        ann_file='annotations/panoptic_val.json',
        data_prefix=dict(img='val_image/', seg='val_panoptic_image/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
val_evaluator_GlaS = dict(
        type='NucleiSegMetric',
        ann_file=_data_root + 'annotations/panoptic_val.json',
        seg_prefix=_data_root + 'val_panoptic_image/',
        outfile_prefix=_data_root + 'tmp/results',
        Comp_Sem = True,
        backend_args=backend_args)

_data_root = f"{root_p}/Folded_Data/Processed_Data_PGDON_D_{fold}/"
dataset_DigestPath = dict(
        type=dataset_type,
        data_root=_data_root,
        metainfo=metainfo,
        return_classes=True,
        ann_file='annotations/panoptic_val.json',
        data_prefix=dict(img='val_image/', seg='val_panoptic_image/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
val_evaluator_DigestPath = dict(
        type='NucleiSegMetric',
        ann_file=_data_root + 'annotations/panoptic_val.json',
        seg_prefix=_data_root + 'val_panoptic_image/',
        outfile_prefix=_data_root + 'tmp/results',
        Comp_Sem = True,
        backend_args=backend_args)

_data_root = f"{root_p}/Folded_Data/Processed_Data_PGDON_O_{fold}/"
dataset_OmniSeg = dict(
        type=dataset_type,
        data_root=_data_root,
        metainfo=metainfo,
        return_classes=True,
        ann_file='annotations/panoptic_val.json',
        data_prefix=dict(img='val_image/', seg='val_panoptic_image/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
val_evaluator_OmniSeg = dict(
        type='NucleiSegMetric',
        ann_file=_data_root + 'annotations/panoptic_val.json',
        seg_prefix=_data_root + 'val_panoptic_image/',
        outfile_prefix=_data_root + 'tmp/results',
        Comp_Sem = True,
        backend_args=backend_args)

_data_root = f"{root_p}/Folded_Data/Processed_Data_PGDON_N_{fold}/"
dataset_NuInsSeg = dict(
        type=dataset_type,
        data_root=_data_root,
        metainfo=metainfo,
        return_classes=True,
        ann_file='annotations/panoptic_val.json',
        data_prefix=dict(img='val_image/', seg='val_panoptic_image/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
val_evaluator_NuInsSeg = dict(
        type='NucleiSegMetric',
        ann_file=_data_root + 'annotations/panoptic_val.json',
        seg_prefix=_data_root + 'val_panoptic_image/',
        outfile_prefix=_data_root + 'tmp/results',
        Comp_Sem = True,
        backend_args=backend_args)

datasets = [
    dataset_DigestPath, dataset_GlaS, dataset_OmniSeg, dataset_PanNuke, dataset_NuInsSeg
]
metrics = [
    val_evaluator_DigestPath, val_evaluator_GlaS, val_evaluator_OmniSeg, val_evaluator_PanNuke, val_evaluator_NuInsSeg
]

# datasets = [
#     dataset_DigestPath
# ]
# metrics = [
#     val_evaluator_DigestPath
# ]


val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(_delete_=True, type='ConcatDataset', datasets=datasets))
test_dataloader = val_dataloader

dataset_prefixes = ['DigestPath', 'GlaS', 'OmniSeg', 'PanNuke', 'NuInsSeg']
# dataset_prefixes = ['DigestPath']
val_evaluator = dict(
    _delete_=True,
    type='MultiDatasetsEvaluator',
    metrics=metrics,
    dataset_prefixes=dataset_prefixes)

test_evaluator = val_evaluator


# learning policy
max_iters = 368750
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[327778, 355092],
    gamma=0.1)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
interval = 2000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=3,
        interval=interval,
        save_best='Avg_DICE',
        rule = 'greater'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

auto_scale_lr = dict(enable=False, base_batch_size=8)
find_unused_parameters = True if multi_head else False


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
)