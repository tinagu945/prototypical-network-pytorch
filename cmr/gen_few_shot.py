"""
Demo of CMR.

Note that CMR assumes that the object has been detected, so please use a picture of a bird that is centered and well cropped.

Sample usage:

python demo.py --name bird_net --num_train_epoch 500 --img_path misc/demo_data/img1.jpg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import sys
# sys.path.insert(0, '/home/zg45/prototypical-network-pytorch/cmr/')
# sys.path.insert(0, '/home/zg45/prototypical-network-pytorch/cmr/nnutils')
# sys.path.insert(0, '/home/zg45/prototypical-network-pytorch/cmr/utils')

from absl import flags, app
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import os

import torch

from nnutils import test_utils
from nnutils import predictor as pred_util
from utils import image as img_util


# flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')

opts = flags.FLAGS
opts.name='bird_net'
opts.num_train_epoch=500


def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

    # Scale the max image size to be img_size
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)

    # Crop img_size x img_size from the center
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # img center in (x, y)
    center = center[::-1]
    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])

    img = img_util.crop(img, bbox, bgval=1.)

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))

    return img


def visualize(img, img_path, outputs, renderer):
    vert = outputs['verts'][0]
    cam = outputs['cam_pred'][0]
    texture = outputs['texture'][0]
    shape_pred = renderer(vert, cam)
    img_pred = renderer(vert, cam, texture=texture)

    # Different viewpoints.
    vp1 = renderer.diff_vp(
        vert, cam, angle=30, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp2 = renderer.diff_vp(
        vert, cam, angle=60, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp3 = renderer.diff_vp(
        vert, cam, angle=60, axis=[1, 0, 0], texture=texture)
    vps=[vp1,vp2,vp3]

    img = np.transpose(img, (1, 2, 0)) 
    
    for i in range(len(vps)):
        plt.imshow(vps[i])
        plt.show()
        plt.savefig(img_path.split('.jpg')[0]+'_vp'+str(i)+'.png')
        
    


def main(_):
    data_folder='/home/zg45/prototypical-network-pytorch/cmr/misc/CUB_200_2011/images/'
    
    predictor = pred_util.MeshPredictor(opts)
    # This is resolution
    renderer = predictor.vis_rend
    renderer.set_light_dir([0, 1, -1], 0.4)

    for cls in sorted(os.listdir(data_folder)):
        for img_path in sorted(os.listdir(os.path.join(data_folder,cls))):
            if 'vp' in img_path:
                continue
            image_path =os.path.join(data_folder,cls,img_path)
            img = preprocess_image(image_path, img_size=opts.img_size)
            print('doing '+image_path)
            batch = {'img': torch.Tensor(np.expand_dims(img, 0))}
            outputs = predictor.predict(batch)

            visualize(img, image_path, outputs, predictor.vis_rend)


if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
