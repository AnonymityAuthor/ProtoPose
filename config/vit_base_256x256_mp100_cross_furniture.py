# model
pose_model = dict(
    type='ProtoPose',
    encoder=dict(
        type='ViT',
        img_size=(256, 256),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        embed_ratio=1,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_scale=True,
        pre_weights='pretrained/dinov2_vitb14_pretrain.pth'
    ),
    decoder=dict(
        type='HeatmapDecoder',
        embed_dim=768,
        upsample_ratio=4
    )
)

# data configuration
data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    use_nms=True,
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    det_bbox_thr=0.0
)

# test config
eval_cfg = dict(
    workers=4,
    flip_test=True,
    post_process='default',
    modulate_kernel=11,
    use_udp=True,
    metric=dict(item='PCK', pck_thr=0.2)
)

# pre-process
train_pipeline = [
    dict(type='LoadImageFromFile', channel_order='bgr'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownRandomTranslation', trans_prob=0.5),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='TopDownGenerateTarget', sigma=2, encoding='UDP', target_type='GaussianHeatmap'),
    dict(type='TopDownGenerateTargetRegression'),
    dict(type='NormalizeImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Collect',
         keys=['image', 'target', 'target_weight', 'joints'],
         meta_keys=['prompt_embedding_info', 'category'])
]

val_pipeline = [
    dict(type='LoadImageFromFile', channel_order='bgr'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='NormalizeImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Collect',
         keys=['image'],
         meta_keys=['image_file', 'center', 'scale', 'flip_index', 'bbox_score', 'bbox_id', 'ep_id',
                    'prompt_embedding_info', 'category', 'eval_mask'])
]

# dataset
data_root = 'YOUR_DATA_DIR'
set_cfg = dict(
    workers_per_gpu=4,
    train=dict(
        type='MP100',
        ann_file='{}/mp100/annotations/mp100_cross_furniture_train.json'.format(data_root),
        img_prefix='{}/mp100/images/'.format(data_root),
        pipeline=train_pipeline,
        num_samples_per_task=6
    ),
    val=dict(
        type='MP100',
        ann_file='{}/mp100/annotations/mp100_cross_furniture_val.json'.format(data_root),
        img_prefix='{}/mp100/images/'.format(data_root),
        pipeline=val_pipeline,
        fsl_pipeline=train_pipeline,
        num_shots=1,
        num_qrys=15,
        num_episodes=100
    ),
    test=dict(
        type='MP100',
        ann_file='{}/mp100/annotations/mp100_cross_furniture_val.json'.format(data_root),
        img_prefix='{}/mp100/images/'.format(data_root),
        pipeline=val_pipeline,
        fsl_pipeline=train_pipeline,
        num_shots=1,
        num_qrys=15,
        num_episodes=200
    )
)

# trainer
trainer = dict(
    pose_model=pose_model,
    criterion=dict(
        mse_loss=dict(type='JointMSELoss', use_target_weight=True, use_target_average=True, weight=0.5)
    ),
    prompt_embeds_cfg=dict(
        human_hand=(21, 768),
        human_face=(71, 768),
        human_body=(19, 768),
        mammal_body=(27, 768),
        bird_body=(15, 768),
        insect_body=(37, 768),
        animal_face=(9, 768),
        vehicle=(13, 768),
        furniture=(21, 768),
        clothes=(55, 768),
    ),
    freeze_encoder_in_ft=True
)

# solver
solver = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.1,
        constructor='LayerDecayOptimizerConstructor',
        paramwise_cfg=dict(
            num_layers=12,
            layer_decay_rate=0.75,
            custom_keys=dict(
                bias=dict(decay_multi=0.),
                pos_embed=dict(decay_mult=0.),
                norm=dict(decay_mult=0.)
            )
        )
    ),
    ft_optimizer=dict(
        type='AdamW',
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.1,
        constructor='LayerDecayOptimizerConstructor',
        paramwise_cfg=dict(
            num_layers=12,
            layer_decay_rate=0.75,
            custom_keys=dict(
                bias=dict(decay_multi=0.),
                pos_embed=dict(decay_mult=0.),
                norm=dict(decay_mult=0.)
            )
        )
    ),
    lr_scheduler=dict(
        warmup_iters=500,
        milestones=[10, 10],
        gamma=0.1
    ),
    total_epochs=10,
    eval_interval={10: 1},   # epoch
    log_interval=25,   # iter
    log_loss=['mse_loss']
)
