import keras
import tensorflow as tf
import numpy as np

from .utils import batch_slice
from .utils import apply_box_deltas_graph
from .utils import clip_boxes_graph

class ProposalLayer(keras.engine.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, max_proposals, nms_threshold, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.max_proposals = max_proposals
        self.nms_threshold = nms_threshold
        self.images_per_gpu = 1
        self.bounding_box_std_dev = [0.1, 0.1, 0.2, 0.2]
        self.pre_nms_max_proposals = 6000
        self.anchors = np.load("Data/anchors.npy")

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.bounding_box_std_dev, [1, 1, 4])

        # Anchors
        anchors = np.broadcast_to(self.anchors, (1,) + self.anchors.shape)

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.pre_nms_max_proposals, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.images_per_gpu)
        deltas = batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.images_per_gpu)
        pre_nms_anchors = batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.images_per_gpu,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),
                                  self.images_per_gpu,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.images_per_gpu,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.max_proposals,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.max_proposals - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = batch_slice([boxes, scores], nms,
                                      self.images_per_gpu)
        proposals = keras.layers.Reshape((self.max_proposals,4))(proposals)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.max_proposals, 4)