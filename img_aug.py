import imgaug.augmenters as iaa
import cv2 as cv
import numpy as np
import imgaug as ia
from PIL import Image
import imageio
import matplotlib.pyplot as plt 
class ImgAug():
    def __init__(self, cfg):
        """Image augmentation
        Args:
            cfg (list[pipeline]):
        """
        self.pipeline = self._compose_pipeline(self,cfg['composer_cfg'], cfg['augmentor_cfg'])

    @staticmethod
    def _name_to_augmentor(name):
        return ''.join(i.capitalize() for i in name.split('_'))

    @staticmethod
    def _compose_pipeline(self,composer_cfg, cfg):
        aug_seq = []
        for i in cfg:
            augmentor_name = i.pop('type')
            augmentor = getattr(iaa, self._name_to_augmentor(augmentor_name))
            aug_seq.append(augmentor(**i))
        composer_name = composer_cfg.pop('type')
        composer = getattr(iaa, self._name_to_augmentor(composer_name))
        return composer(aug_seq, **composer_cfg)

    def __call__(self,x):
        return self.pipeline(image=x)
        

def test_imgaug():
    cfg = dict(
        composer_cfg=dict(type='one_of'),
        augmentor_cfg=[
            dict(type='affine', rotate=10),
            dict(type='additive_gaussian_noise', scale=0.1*255)
#             dict(type='Add',n=50, per_channel=True)
#     dict(iaa.Sharpen(alpha=0.5))
        ]
    )
    augmentor = ImgAug(cfg)
    image = imageio.imread(filepath)
#     image = np.array(
#     ia.quokka(size=(64, 64)),
#     dtype=np.uint8)
    img_aug = augmentor(image)
#     ia.imshow(img_aug)
    plt.imshow(img_aug)
    print(type(img_aug))
    return img_aug

#     return image


if __name__ == '__main__':
    test_imgaug()
