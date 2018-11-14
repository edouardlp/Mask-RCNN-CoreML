import keras
import tensorflow as tf

class RPNGraph():

    anchor_stride = 1
    anchors_per_location = len([0.5, 1, 2])
    depth = 256

    def build(self):

        anchor_stride = self.anchor_stride
        anchors_per_location = self.anchors_per_location

        feature_map = keras.layers.Input(shape=[None, None, self.depth],
                                 name="input_rpn_feature_map")

        shared = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=anchor_stride,
                           name='rpn_conv_shared')(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = keras.layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)

        # Reshape to [batch, anchors, 2]
        rpn_class_logits = keras.layers.Reshape((-1, 2))(x)


        # Softmax on last dimension of BG/FG.
        rpn_class = keras.layers.Activation(
            "softmax", name="rpn_class_xxx")(rpn_class_logits)


        # Bounding box refinement. [batch, H, W, anchors per location * depth]
        # where depth is [x, y, log(w), log(h)]
        x = keras.layers.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                      activation='linear', name='rpn_bbox_pred')(shared)

        # Reshape to [batch, anchors, 4]
        rpn_bbox = keras.layers.Reshape((-1, 4))(x)

        return keras.models.Model(inputs=[feature_map], outputs=[rpn_class, rpn_bbox], name="rpn_model")
