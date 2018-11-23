import argparse
import json
import os
import skimage
import numpy as np

import model

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
        '--images_path',
        help='Path to images',
        default="Data/images",
        required=False
    )

    args = parser.parse_args()
    params = args.__dict__
    
    config_path = params.pop('config_path')
    weights_path = params.pop('weights_path')
    results_path = params.pop('results_path')
    images_path = params.pop('images_path')

    images = []

    for filename in os.listdir(images_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = skimage.io.imread(os.path.join(images_path, filename))
            images.append(image)

    model.predict(config_path=config_path,
                  weights_path=weights_path,
                  results_path=results_path,
                  images=np.array(images),
                  params=params)
