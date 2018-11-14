import argparse
import json
import os

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
        '--anchors_path',
        help='Path to anchors file',
        default="Data/anchors.npy",
        required=False
    )

    parser.add_argument(
        '--export_main_path',
        help='Path to export main file',
        default="Data/MaskRCNN.mlmodel",
        required=False
    )

    parser.add_argument(
        '--export_mask_path',
        help='Path to export mask file',
        default="Data/Mask.mlmodel",
        required=False
    )

    parser.add_argument(
        '--export_anchors_path',
        help='Path to export anchors file',
        default="Data/anchors.bin",
        required=False
    )

    args = parser.parse_args()
    params = args.__dict__
    
    config_path = params.pop('config_path')
    weights_path = params.pop('weights_path')
    #TODO: remove and generate instead
    anchors_path = params.pop('anchors_path')
    export_main_path = params.pop('export_main_path')
    export_mask_path = params.pop('export_mask_path')
    #TODO: remove and generate instead
    export_anchors_path = params.pop('export_anchors_path')

    model.export(config_path=config_path,
                 weights_path=weights_path,
                 anchors_path=anchors_path,
                 export_main_path=export_main_path,
                 export_mask_path=export_mask_path,
                 export_anchors_path=export_anchors_path,
                 params=params)
