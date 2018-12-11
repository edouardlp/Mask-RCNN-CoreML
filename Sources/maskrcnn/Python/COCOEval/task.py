import argparse
import json

from maskrcnn.model import Config
from maskrcnn.model import MaskRCNNModel
from maskrcnn.datasets.coco import COCODataset
from maskrcnn import results_pb2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path',
                        help='Path to config file',
                        default="model/config.json",
                        required=False)

    parser.add_argument('--weights_path',
                        help='Path to weights file',
                        default="model/weights.h5",
                        required=False)

    parser.add_argument('--model_dir',
                        help='Path to model dir',
                        default="products/model_dir",
                        required=False)

    parser.add_argument('--results_path',
                        help='Path to results',
                        required=True)

    parser.add_argument('--dataset_path',
                        help='Path to dataset',
                        default="data/coco/data.tfrecords",
                        required=False)

    parser.add_argument('--coco_annotations_dir',
                        help='Path to COCO annotations directory',
                        default="data/coco",
                        required=False)

    parser.add_argument('--coco_images_dir',
                        help='Path to COCO images directory',
                        default="data/coco/val2017",
                        required=False)

    parser.add_argument('--coco_year',
                        help='COCO Year',
                        default="2017",
                        required=False)

    parser.add_argument('--coco_type',
                        help='COCO Type',
                        default="val",
                        required=False)

    parser.add_argument('--compare',
                        help='Compare to model',
                        action='store_true')

    args = parser.parse_args()
    params = args.__dict__

    config_path = params.pop('config_path')
    weights_path = params.pop('weights_path')
    model_dir = params.pop('model_dir')
    results_path = params.pop('results_path')
    dataset_path = params.pop('dataset_path')
    coco_annotations_dir = params.pop('coco_annotations_dir')
    coco_images_dir = params.pop('coco_images_dir')
    compare = params.pop('compare')

    coco_year = params.pop('coco_year')
    coco_type = params.pop('coco_type')
    print(coco_year)
    print(coco_type)
    print(results_path)
    config = Config()
    with open(config_path) as file:
        config_dict = json.load(file)
        config.__dict__.update(config_dict)

    model = MaskRCNNModel(config_path,
                      model_dir=model_dir,
                      initial_keras_weights=weights_path)

    coco_dataset = COCODataset(path=dataset_path,
                           type=coco_type,
                           year=coco_year,
                           annotations_dir=coco_annotations_dir,
                           images_dir=coco_images_dir,
                           image_shape=(config.input_width, config.input_height, 3))

    input_results = results_pb2.Results()
    f = open(results_path, "rb")
    input_results.ParseFromString(f.read())
    f.close()
    print("Printing CoreML results")
    coco_dataset.evaluate_results(input_results)
    if compare:
        coco_dataset.preprocess(reprocess_if_exists=True, limit=5)
        input_fn = coco_dataset.make_input_fn(batch_size=1, limit=5)
        results = model.predict(dataset_id=coco_dataset.id, input_fn=input_fn,
                            class_label_fn=coco_dataset.class_label_from_id)
        print("Printing Keras results")
        coco_dataset.evaluate_results(results)
