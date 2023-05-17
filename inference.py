from unittest import result
from mmseg.apis import init_model
from mmseg.apis.inference import inference_model, show_result_pyplot
from PIL import Image
import numpy as np

config_path = 'my_config/deeplabv3plus_pretrain.py'
checkpoint_path = 'checkpoints/iter_2000.pth'
img_path = 'data/VOCdevkit/VOC2012/JPEGImages/2008_000034.jpg'

model = init_model(config_path, checkpoint_path, device='cuda:0')
result = inference_model(model, img_path)
show_result_pyplot(model, img_path, result, out_file='inference/result_deeplab_pretrain.png', show=False, opacity=0.8)

