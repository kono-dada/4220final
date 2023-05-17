_base_ = [
    'fcn_unet_s5-d16.py',
    'pascal_voc12.py', 'default_runtime.py',
    'schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=7),
    auxiliary_head=dict(num_classes=7))