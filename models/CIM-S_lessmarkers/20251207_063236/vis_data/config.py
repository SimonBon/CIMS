batch_size = 512
classes_to_ignore = [
    'Seg Artifact',
]
crops = [
    20,
    18,
]
custom_hooks = [
    dict(epochs=20, priority='VERY_LOW', type='EvaluateModel'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'MIDL26.MIDL26.src.SimCLR',
        'MIDL26.MIDL26.src._dataset',
        'MIDL26.MIDL26.src._transforms',
        'MIDL26.MIDL26.src._nextmarker_SE',
        'MIDL26.MIDL26.src._lrp_backbones',
        'MIDL26.MIDL26.src._classifier_hook',
        'MIDL26.MIDL26.src.LRPModel',
    ])
data_preprocessor = None
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=50, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=1, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmselfsup'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
feature_dim = 72
h5_file_path = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/cHL_data/h5/cHL_CODEX.h5'
image_key = 'NORMED_IMAGES'
in_channels = 18
launcher = 'none'
layers = [
    1,
]
load_from = None
log_level = 'INFO'
log_processor = dict(
    custom_cfg=[
        dict(data_src='', method='mean', window_size='global'),
    ],
    window_size=1)
lr = 0.3
markers_to_use = [
    'DAPI-01',
    'CD11b',
    'CD11c',
    'CD15',
    'CD163',
    'CD20',
    'CD206',
    'CD30',
    'CD31',
    'CD4',
    'CD56',
    'CD68',
    'CD7',
    'CD8',
    'Cytokeritin',
    'FoxP3',
    'MCT',
    'Podoplanin',
]
mask_image = False
model = dict(
    backbone=dict(
        arctype='ParallelResNet',
        backbone_kwargs=dict(
            in_channels=18, layers=[
                1,
            ], width_per_channel=4),
        in_channels=18,
        type='LRPModel'),
    data_preprocessor=None,
    head=dict(
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.2,
        type='ContrastiveHead'),
    neck=dict(
        hid_channels=128,
        in_channels=72,
        num_layers=2,
        out_channels=128,
        type='NonLinearNeck',
        with_avg_pool=False),
    type='SimCLR2')
n_cosine = 2800
n_linear = 200
optim_wrapper = dict(
    optimizer=dict(lr=0.3, momentum=0.9, type='LARS', weight_decay=1e-05),
    type='OptimWrapper')
optimizer = dict(lr=0.3, momentum=0.9, type='LARS', weight_decay=1e-05)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=200, start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=2800,
        begin=200,
        by_epoch=False,
        end=3000,
        eta_min=0.03,
        type='CosineAnnealingLR'),
]
resume = False
train_cfg = dict(max_iters=3000, type='IterBasedTrainLoop')
train_dataloader = dict(
    batch_size=512,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        classes_to_ignore=[
            'Seg Artifact',
        ],
        h5_file_path=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/cHL_data/h5/cHL_CODEX.h5',
        image_key='NORMED_IMAGES',
        in_memory=True,
        markers_to_use=[
            'DAPI-01',
            'CD11b',
            'CD11c',
            'CD15',
            'CD163',
            'CD20',
            'CD206',
            'CD30',
            'CD31',
            'CD4',
            'CD56',
            'CD68',
            'CD7',
            'CD8',
            'Cytokeritin',
            'FoxP3',
            'MCT',
            'Podoplanin',
        ],
        mask_image=False,
        patch_size=40,
        pipeline=[
            dict(
                num_views=[
                    1,
                    1,
                ],
                transforms=[
                    [
                        dict(
                            horizontal=True,
                            prob=0.5,
                            type='C_RandomFlip',
                            vertical=True),
                        dict(
                            angle=(
                                0,
                                360,
                            ),
                            order=1,
                            scale=(
                                0.8,
                                1.2,
                            ),
                            shift=(
                                0,
                                0,
                            ),
                            type='C_RandomAffine'),
                        dict(
                            clip=True,
                            high=1.2,
                            low=0.9,
                            type='RandomIntensity'),
                        dict(
                            clip=True,
                            mean=(
                                0,
                                0,
                            ),
                            std=(
                                0,
                                0.02,
                            ),
                            type='RandomNoise'),
                        dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                        dict(size=20, type='CentralCutter'),
                    ],
                    [
                        dict(
                            horizontal=True,
                            prob=0.5,
                            type='C_RandomFlip',
                            vertical=True),
                        dict(
                            angle=(
                                0,
                                360,
                            ),
                            order=1,
                            scale=(
                                0.8,
                                1.2,
                            ),
                            shift=(
                                0,
                                0,
                            ),
                            type='C_RandomAffine'),
                        dict(
                            clip=True,
                            high=1.2,
                            low=0.9,
                            type='RandomIntensity'),
                        dict(
                            clip=True,
                            mean=(
                                0,
                                0,
                            ),
                            std=(
                                0,
                                0.02,
                            ),
                            type='RandomNoise'),
                        dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                        dict(size=18, type='CentralCutter'),
                    ],
                ],
                type='MultiView'),
            dict(type='PackSelfSupInputs'),
        ],
        split=[
            0.7,
            0.1,
            0.2,
        ],
        type='MultiChannelDataset',
        used_split='training'),
    drop_last=True,
    num_workers=32,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        num_views=[
            1,
            1,
        ],
        transforms=[
            [
                dict(
                    horizontal=True,
                    prob=0.5,
                    type='C_RandomFlip',
                    vertical=True),
                dict(
                    angle=(
                        0,
                        360,
                    ),
                    order=1,
                    scale=(
                        0.8,
                        1.2,
                    ),
                    shift=(
                        0,
                        0,
                    ),
                    type='C_RandomAffine'),
                dict(clip=True, high=1.2, low=0.9, type='RandomIntensity'),
                dict(
                    clip=True,
                    mean=(
                        0,
                        0,
                    ),
                    std=(
                        0,
                        0.02,
                    ),
                    type='RandomNoise'),
                dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                dict(size=20, type='CentralCutter'),
            ],
            [
                dict(
                    horizontal=True,
                    prob=0.5,
                    type='C_RandomFlip',
                    vertical=True),
                dict(
                    angle=(
                        0,
                        360,
                    ),
                    order=1,
                    scale=(
                        0.8,
                        1.2,
                    ),
                    shift=(
                        0,
                        0,
                    ),
                    type='C_RandomAffine'),
                dict(clip=True, high=1.2, low=0.9, type='RandomIntensity'),
                dict(
                    clip=True,
                    mean=(
                        0,
                        0,
                    ),
                    std=(
                        0,
                        0.02,
                    ),
                    type='RandomNoise'),
                dict(drop_prob=0.1, type='C_RandomChannelDrop'),
                dict(size=18, type='CentralCutter'),
            ],
        ],
        type='MultiView'),
    dict(type='PackSelfSupInputs'),
]
val_crop = 18
val_pipeline = [
    dict(size=18, type='CentralCutter'),
]
view_pipelines = [
    [
        dict(horizontal=True, prob=0.5, type='C_RandomFlip', vertical=True),
        dict(
            angle=(
                0,
                360,
            ),
            order=1,
            scale=(
                0.8,
                1.2,
            ),
            shift=(
                0,
                0,
            ),
            type='C_RandomAffine'),
        dict(clip=True, high=1.2, low=0.9, type='RandomIntensity'),
        dict(clip=True, mean=(
            0,
            0,
        ), std=(
            0,
            0.02,
        ), type='RandomNoise'),
        dict(drop_prob=0.1, type='C_RandomChannelDrop'),
        dict(size=20, type='CentralCutter'),
    ],
    [
        dict(horizontal=True, prob=0.5, type='C_RandomFlip', vertical=True),
        dict(
            angle=(
                0,
                360,
            ),
            order=1,
            scale=(
                0.8,
                1.2,
            ),
            shift=(
                0,
                0,
            ),
            type='C_RandomAffine'),
        dict(clip=True, high=1.2, low=0.9, type='RandomIntensity'),
        dict(clip=True, mean=(
            0,
            0,
        ), std=(
            0,
            0.02,
        ), type='RandomNoise'),
        dict(drop_prob=0.1, type='C_RandomChannelDrop'),
        dict(size=18, type='CentralCutter'),
    ],
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SelfSupVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
width_per_channel = 4
work_dir = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/MIDL26/work_dir/Best_ParallelResNet_less_markers'
