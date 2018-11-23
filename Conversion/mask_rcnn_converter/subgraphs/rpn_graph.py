import keras

class RPNGraph():

    def __init__(self,
                 anchor_stride,
                 anchors_per_location,
                 depth,
                 feature_maps):
        self.anchor_stride = anchor_stride
        self.anchors_per_location = anchors_per_location
        self.depth = depth
        self.feature_maps = feature_maps

    def build(self):

        anchor_stride = self.anchor_stride
        anchors_per_location = self.anchors_per_location
        depth = self.depth

        feature_map = keras.layers.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")

        shared = keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                           strides=anchor_stride,
                           name='rpn_conv_shared')(feature_map)

        # Anchor Score. [batch, height, width, anchors per location * 2].
        x = keras.layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                      activation='linear', name='rpn_class_raw')(shared)

        #TODO: this appears to break CoreML on iPhone XS

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

        model = keras.models.Model(inputs=[feature_map], outputs=[rpn_class, rpn_bbox], name="rpn_model")

        layer_outputs = []
        for p in self.feature_maps:
            layer_outputs.append(model([p]))
        output_names = ["rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [keras.layers.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class, rpn_bbox = outputs

        return rpn_class, rpn_bbox
