import keras
import tensorflow as tf
import numpy as np

from .utils import batch_slice
from .utils import apply_box_deltas_graph
from .utils import clip_boxes_graph
from .utils import norm_boxes_graph

#NOTE: None of this will get exported to CoreML. This is only useful for python inference, and for CoreML to determine
#input and output shapes.

def refine_detections_graph(rois,
                            classifications,
                            window,
                            BBOX_STD_DEV,
                            DETECTION_MIN_CONFIDENCE,
                            DETECTION_MAX_INSTANCES,
                            DETECTION_NMS_THRESHOLD):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.to_int32(classifications[:,4])
    deltas_specific = classifications[:,0:4]
    # Class probability of the top class of each ROI
    class_scores = classifications[:,5] #index 5
    # Class-specific bounding box deltas
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=DETECTION_MAX_INSTANCES,
                iou_threshold=DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    # Keep top detections
    roi_count = DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return tf.reshape(detections, (-1,6))


class DetectionLayer(keras.engine.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self,
                 max_detections,
                 bounding_box_std_dev,
                 detection_min_confidence,
                 detection_nms_threshold, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        #TODO: since this is inference only, we may want to remove this
        self.images_per_gpu = 1
        self.batch_size = 1
        self.max_detections = max_detections
        self.bounding_box_std_dev = bounding_box_std_dev
        self.detection_min_confidence = detection_min_confidence
        self.detection_nms_threshold = detection_nms_threshold

    def call(self, inputs):
        rois = inputs[0]
        classifications = inputs[1]
        #mrcnn_bbox = inputs[2]
        #mrcnn_bbox = keras.layers.Permute((3, 2, 1))(mrcnn_bbox)
        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.

        #TODO: get these sizes from config
        image_shape = tf.convert_to_tensor([256,256,3], dtype=tf.float32)
        window = tf.convert_to_tensor([0,0,256,256], dtype=tf.float32)

        window = tf.reshape(window, [-1,4])
        window = norm_boxes_graph(window, image_shape[:2])
        window = tf.reshape(window, [-1,4])

        #window = tf.reshape(window, (None,4))
        # Run detection refinement graph on each item in the batch
        detections_batch = batch_slice(
            [rois, classifications, window],
            lambda x, y, z: refine_detections_graph(x, y, z, np.array(self.bounding_box_std_dev), self.detection_min_confidence, self.max_detections, self.detection_nms_threshold),
            self.images_per_gpu)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        detections = tf.reshape(
            detections_batch,
            [self.batch_size, self.max_detections, 6])
        return detections

    def compute_output_shape(self, input_shape):
        return (None, 1, self.max_detections, 6)
