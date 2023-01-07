_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

CLASSES = (
    'building',
)
model = dict(
    backbone=dict(in_channels=4),
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(CLASSES)),
        mask_head=dict(
            num_classes=len(CLASSES))
    ),
    test_cfg=dict(
        rcnn=dict(max_per_img=300),
    )
)

img_norm_cfg = dict(
    mean=[0., 0., 0., 0.], std=[255., 255., 255., 255.], to_rgb=False)

dataset_type = 'CocoDataset'
data_root = '/workspace/dataset/dfc2023_track1/'
SIZE = [(512, 512), (864, 864)]
flip_ratio = 0.5
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=SIZE, keep_ratio=True, multiscale_mode='range'),
    dict(type='RandomFlip', flip_ratio=flip_ratio),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(768, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=flip_ratio),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.3,
        dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/buildings_only_train.json',
        img_prefix=data_root + 'train/merge/',
        pipeline=train_pipeline,
        classes=CLASSES
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/buildings_only_val.json',
        img_prefix=data_root + 'val/merge/',
        pipeline=test_pipeline,
        classes=CLASSES),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'image_id/image_id_val.json',
        img_prefix=data_root + 'val/merge/',
        pipeline=test_pipeline,
        classes=CLASSES))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[24, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
auto_scale_lr = dict(enable=True, base_batch_size=16)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
