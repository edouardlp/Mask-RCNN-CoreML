from maskrcnn.model import MaskRCNNModel
from maskrcnn.datasets.coco import COCODataset

model = MaskRCNNModel("", model_dir="Data/model/", initial_keras_weights="Data/weights.h5")

coco_dataset = COCODataset(path="Data/coco/data.tfrecords",
                           type='val',
                           year='2017',
                           annotations_dir="Data/coco_annotations",
                           images_dir="Data/val2017",
                           image_shape=(1024,1024,3))
print("CoreML Results")
print("WIP")

coco_dataset.preprocess(reprocess_if_exists=True, limit=5)
input_fn = coco_dataset.make_input_fn(batch_size=1, limit=5)
results = model.predict(dataset_id=coco_dataset.id,input_fn=input_fn,class_label_fn=coco_dataset.class_label_from_id)

print("Keras Results")
coco_dataset.evaluate_results(results)
