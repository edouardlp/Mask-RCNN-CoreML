import keras
import tensorflow as tf
import numpy as np

from .pyramid_roi_align_layer import PyramidROIAlign

#We use a custom layer to implement the time distributed layer in Swift
class TimeDistributedMask(keras.layers.Layer):

    def __init__(self, max_regions, num_classes, depth, **kwargs):
        super(TimeDistributedMask, self).__init__(**kwargs)
        self.timesteps = max_regions
        self.num_classes = num_classes
        self.depth = depth

    #This will not get exported to CoreML
    def _build_keras_inner_model(self):

        num_classes = self.num_classes
        depth = self.depth

        #TODO: use time distributed here

        input = keras.layers.Input((self.pool_size, self.pool_size, depth))

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv1")(input)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn1')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv2")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn2')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv3")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn3')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv4")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn4')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu", name="mrcnn_mask_deconv")(x)
        x = keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid", name="mrcnn_mask")(x)
        x = keras.layers.Reshape((self.pool_size * 2, self.pool_size * 2, num_classes))(x)

        return keras.models.Model(inputs=[input], outputs=[x])

    def call(self, inputs):
        pyramid = inputs[0]
        rois = inputs[1]
        #TODO: call inner model
        #TODO: extract only the relevant class
        return tf.convert_to_tensor(np.zeros((28, 28,self.timesteps)), dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return (None,28, 28, self.timesteps)


class FPNMaskGraph():

    def __init__(self,
                 rois,
                 feature_maps,
                 pool_size,
                 image_shape,
                 num_classes,
                 max_regions,
                 pyramid_top_down_size,
                 weights_path):

        self.rois = rois
        self.feature_maps = feature_maps
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.max_regions = max_regions
        self.pyramid_top_down_size = pyramid_top_down_size
        self.weights_path = weights_path

    def _build_coreml_inner_model(self):

        num_classes = self.num_classes
        pyramid_top_down_size = self.pyramid_top_down_size

        input = keras.layers.Input((self.pool_size, self.pool_size, pyramid_top_down_size))

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv1")(input)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn1')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv2")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn2')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv3")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn3')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv4")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn4')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu", name="mrcnn_mask_deconv")(x)
        x = keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid", name="mrcnn_mask")(x)
        x = keras.layers.Reshape((self.pool_size * 2, self.pool_size * 2, num_classes))(x)

        return keras.models.Model(inputs=[input], outputs=[x])

    def build(self):

        rois = self.rois
        feature_maps = self.feature_maps

        pool_size = self.pool_size
        image_shape = self.image_shape
        num_classes = self.num_classes
        max_regions = self.max_regions
        pyramid_top_down_size = self.pyramid_top_down_size

        pyramid = PyramidROIAlign(name="roi_align_mask",
                                  pool_shape=[pool_size, pool_size],
                                  image_shape=image_shape)([rois] + feature_maps)

        x = TimeDistributedMask(max_regions=max_regions, num_classes=num_classes, depth=pyramid_top_down_size)([pyramid, rois])

        pyramid_align_result_size = pool_size * 2
        result = keras.layers.Reshape((pyramid_align_result_size, pyramid_align_result_size, max_regions))(x)
        fpn_mask_model = self._build_coreml_inner_model()
        fpn_mask_model.load_weights(self.weights_path, by_name=True)
        return fpn_mask_model, result