import keras
import tensorflow as tf
import numpy as np

from .pyramid_roi_align_layer import PyramidROIAlign

#We use a custom layer to implement the time distributed layer in Swift
class TimeDistributedClassifier(keras.layers.Layer):

    def __init__(self, max_regions, pool_size, num_classes, pyramid_top_down_size, fc_layers_size, weights_path, **kwargs):
        super(TimeDistributedClassifier, self).__init__(**kwargs)
        self.max_regions = max_regions
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.pyramid_top_down_size = pyramid_top_down_size
        self.fc_layers_size = fc_layers_size
        self.weights_path = weights_path

    def _build_keras_inner_model(self):

        pool_size = self.pool_size
        fc_layers_size = self.fc_layers_size
        pyramid_top_down_size = self.pyramid_top_down_size
        num_classes = self.num_classes

        input = keras.layers.Input((self.max_regions,self.pool_size, self.pool_size, pyramid_top_down_size))

        x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1")(
            input)
        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(),name='mrcnn_class_bn1')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size, (1, 1)),name="mrcnn_class_conv2")(x)
        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(),name='mrcnn_class_bn2')(x, training=False)
        shared = keras.layers.Activation('relu')(x)
        shared = keras.layers.Lambda(lambda x: keras.backend.squeeze(keras.backend.squeeze(x, 3), 2),
                           name="pool_squeeze")(shared)
        logits = keras.layers.TimeDistributed(keras.layers.Dense(num_classes), name='mrcnn_class_logits')(shared)
        probabilities = keras.layers.TimeDistributed(keras.layers.Activation("softmax"), name="mrcnn_class")(logits)
        bounding_boxes = keras.layers.TimeDistributed(keras.layers.Dense(num_classes * 4, activation='linear'), name='mrcnn_bbox_fc')(shared)
        return keras.models.Model(inputs=[input], outputs=[probabilities,bounding_boxes])

    def call(self, inputs):
        model = self._build_keras_inner_model()
        model.load_weights(self.weights_path, by_name=True)
        probabilities, bounding_boxes = model(inputs)
        probabilities = tf.reshape(probabilities, shape=(self.max_regions, self.num_classes))
        bounding_boxes = tf.reshape(bounding_boxes, shape=(self.max_regions, self.num_classes,4))
        class_ids = tf.argmax(probabilities, axis=1, output_type=tf.int32)
        indices = tf.stack([tf.range(probabilities.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probabilities, indices)
        deltas_specific = tf.gather_nd(bounding_boxes, indices)
        class_ids = tf.to_float(class_ids)

        result = tf.concat([deltas_specific,tf.expand_dims(class_ids,axis=1),tf.expand_dims(class_scores,axis=1)], axis=1)
        result = tf.expand_dims(result,axis=0)
        return result

    def compute_output_shape(self, input_shape):
        #(dy,dx,log(dh),log(dw),classId,score)
        return (1,self.max_regions,6)

class FPNClassifierGraph():

    def __init__(self,
                 rois,
                 feature_maps,
                 pool_size,
                 image_shape,
                 num_classes,
                 max_regions,
                 fc_layers_size,
                 pyramid_top_down_size,
                 weights_path):

        self.rois = rois
        self.feature_maps = feature_maps
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.max_regions = max_regions
        self.fc_layers_size = fc_layers_size
        self.pyramid_top_down_size = pyramid_top_down_size
        self.weights_path = weights_path

    def _build_coreml_inner_model(self):

        pool_size = self.pool_size
        fc_layers_size = self.fc_layers_size
        pyramid_top_down_size = self.pyramid_top_down_size
        num_classes = self.num_classes

        input = keras.layers.Input((self.pool_size, self.pool_size, pyramid_top_down_size))

        #TODO: Attempt to use same tricks as MobileNet

        x = keras.layers.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid", name="mrcnn_class_conv1")(
            input)
        x = keras.layers.BatchNormalization(name='mrcnn_class_bn1')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(fc_layers_size, (1, 1),name="mrcnn_class_conv2")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_class_bn2')(x, training=False)
        shared = keras.layers.Activation('relu')(x)
        logits = keras.layers.Dense(num_classes, name='mrcnn_class_logits')(shared)
        probabilities = keras.layers.Activation("softmax", name="mrcnn_class")(logits)
        bounding_boxes = keras.layers.Dense(num_classes * 4, activation='linear', name='mrcnn_bbox_fc')(shared)

        #TODO : reduce probabilities and bounding_boxes

        return keras.models.Model(inputs=[input], outputs=[probabilities,bounding_boxes])

    def build(self):

        rois = self.rois
        feature_maps = self.feature_maps

        pool_size = self.pool_size
        num_classes = self.num_classes
        image_shape = self.image_shape
        max_regions = self.max_regions
        pyramid_top_down_size = self.pyramid_top_down_size
        fc_layers_size = self.fc_layers_size

        pyramid = PyramidROIAlign(name="roi_align_classifier",
                                  pool_shape=[pool_size, pool_size],
                                  image_shape=image_shape)([rois] + feature_maps)

        fpn_classifier_model = self._build_coreml_inner_model()
        fpn_classifier_model.load_weights(self.weights_path, by_name=True)

        classification = TimeDistributedClassifier(max_regions=max_regions, pool_size=pool_size, num_classes=num_classes, pyramid_top_down_size=pyramid_top_down_size, fc_layers_size=fc_layers_size, weights_path=self.weights_path)([pyramid])

        return fpn_classifier_model,classification