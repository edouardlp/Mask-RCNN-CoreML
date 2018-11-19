import keras

from .pyramid_roi_align_layer import PyramidROIAlign

class FPNClassifierGraph():

    def __init__(self,
                 rois,
                 feature_maps,
                 pool_size,
                 image_shape,
                 num_classes,
                 max_regions,
                 fc_layers_size,
                 pyramid_top_down_size):

        self.rois = rois
        self.feature_maps = feature_maps
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.max_regions = max_regions
        self.fc_layers_size = fc_layers_size
        self.pyramid_top_down_size = pyramid_top_down_size

    def build(self):

        rois = self.rois
        feature_maps = self.feature_maps

        pool_size = self.pool_size
        image_shape = self.image_shape
        num_classes = self.num_classes
        max_regions = self.max_regions
        fc_layers_size = self.fc_layers_size
        pyramid_top_down_size = self.pyramid_top_down_size

        pyramid = PyramidROIAlign(name="roi_align_classifier",
                                  pool_shape=[pool_size, pool_size],
                                  image_shape=image_shape)([rois] + feature_maps)

        #We reshape the pyramid to in effect apply a TimeDistributed layer using only Convolutions
        reshaped_pyramid = keras.layers.Reshape((pool_size*max_regions,pool_size,pyramid_top_down_size))(pyramid)

        #Apply a stride of pool_size to treat each roi independently. output: max_regions x 1 x fc_layers_size
        x = keras.layers.Conv2D(fc_layers_size, (pool_size, pool_size), strides=pool_size, padding="valid", name="mrcnn_class_conv1")(
            reshaped_pyramid)
        x = keras.layers.BatchNormalization(name='mrcnn_class_bn1')(x, training=False)
        x = keras.layers.Activation('relu')(x)
        # output: max_regions x 1 x fc_layers_size
        x = keras.layers.Conv2D(fc_layers_size, (1, 1),name="mrcnn_class_conv2")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_class_bn2')(x, training=False)

        shared = keras.layers.Activation('relu')(x)
        shared = keras.layers.Reshape((max_regions,fc_layers_size))(shared)

        logits = keras.layers.Dense(num_classes, name='mrcnn_class_logits')(shared)
        probabilities = keras.layers.Activation("softmax", name="mrcnn_class")(logits)

        bounding_boxes = keras.layers.Dense(num_classes * 4, activation='linear', name='mrcnn_bbox_fc')(shared)
        bounding_boxes = keras.layers.Reshape((max_regions,num_classes,4))(bounding_boxes)
        bounding_boxes = keras.layers.Permute((3, 2, 1))(bounding_boxes)

        return probabilities,bounding_boxes