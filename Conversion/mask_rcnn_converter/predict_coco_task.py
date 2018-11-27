import argparse
import json
import os
import skimage
import numpy as np

import model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config_path',
        help='Path to config file',
        default="Data/config.json",
        required=False
    )

    parser.add_argument(
        '--weights_path',
        help='Path to weights file',
        default="Data/weights.h5",
        required=False
    )

    parser.add_argument(
        '--results_path',
        help='Path to write results',
        default="Data/results.json",
        required=False
    )

    parser.add_argument(
        '--coco_annotations_path',
        help='Path to images',
        default="Data/coco_annotations/instances_val2017.json",
        required=False
    )

    parser.add_argument(
        '--coco_images_path',
        help='Path to images',
        default="Data/coco_images/",
        required=False
    )

    args = parser.parse_args()
    params = args.__dict__
    
    config_path = params.pop('config_path')
    weights_path = params.pop('weights_path')
    results_path = params.pop('results_path')
    coco_annotations_path = params.pop('coco_annotations_path')
    coco_images_path = params.pop('coco_images_path')

    coco = COCO(coco_annotations_path)
    coco_image_ids = coco.getImgIds()
    coco_image_ids = coco_image_ids[0:1]

    print(coco_image_ids)

    image_annotations = coco.loadImgs(coco_image_ids)
    images = []

    for annotation in image_annotations:
        print(annotation)
        image = skimage.io.imread(os.path.join(coco_images_path, annotation["file_name"]))
        images.append(image)

    model.predict(config_path=config_path,
                  weights_path=weights_path,
                  results_path=results_path,
                  image_ids=coco_image_ids,
                  images=np.array(images),
                  params=params)
