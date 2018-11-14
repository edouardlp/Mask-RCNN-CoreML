import keras
import tensorflow as tf
import coremltools

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.log(x) / tf.log(2.0)

class DebugLayer(keras.engine.Layer):

    def __init__(self, **kwargs):
        super(DebugLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

class PyramidROIAlign(keras.engine.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_shape = (1024,1024)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0][:,:,0:4]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[1:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = self.image_shape
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

class FixedTimeDistributed(keras.layers.Layer):

    def __init__(self, model, timesteps, output_names, output_shapes, **kwargs):
        super(FixedTimeDistributed, self).__init__(**kwargs)
        self.model = model
        self.timesteps = timesteps
        self.output_names = output_names
        self.output_shapes = output_shapes

    def call(self, inputs):

        pyramid = inputs[0]
        slice_outputs = []
        #This will not get export to CoreML. We will vectorize this in CoreML
        for i in range(0,self.timesteps):
            pyramid_slice = pyramid[i,:,:,:]
            pyramid_slice = tf.expand_dims(pyramid_slice, 0)
            outputs = self.model([pyramid_slice])
            outputs = tf.expand_dims(outputs[:,:,:,0], 3)
            slice_outputs.append(outputs)

        if len(self.output_names) == 1:
            output = tf.concat((slice_outputs), 3)
            return output

        outputs = list(zip(*slice_outputs))
        outputs = [keras.layers.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, self.output_names)]
        return outputs

    def compute_output_shape(self, input_shape):
        return (None,28, 28, self.timesteps)
        return [tuple([None, self.timesteps] + [o]) for o in self.output_shapes]

class FPNClassifierGraph():

    def __init__(self, rois, feature_maps):
        self.rois = rois
        self.feature_maps = feature_maps

    def build(self):

        pool_size = 7
        rois = self.rois
        feature_maps = self.feature_maps
        num_classes = 80+1
        rois_size = 1000
        fc_layers_size = 1024
        train_bn = False
        pyramid_top_down_size = 256

        pyramid = PyramidROIAlign([pool_size, pool_size],
                            name="roi_align_classifier")([rois] + feature_maps)

        #We reshape the pyramid to in effect apply a TimeDistributed layer using only Convolutions
        reshaped_pyramid = keras.layers.Reshape((pool_size*rois_size,pool_size,pyramid_top_down_size))(pyramid)
        #reshaped_pyramid = DebugLayer()(reshaped_pyramid)

        #Apply a stride of pool_size to treat each roi independently. output: rois_size x 1 x fc_layers_size
        x = keras.layers.Conv2D(fc_layers_size, (pool_size, pool_size), strides=7,padding="valid", name="mrcnn_class_conv1")(
            reshaped_pyramid)
        x = keras.layers.BatchNormalization(name='mrcnn_class_bn1')(x, training=train_bn)
        x = keras.layers.Activation('relu')(x)
        # output: rois_size x 1 x fc_layers_size
        x = keras.layers.Conv2D(fc_layers_size, (1, 1),name="mrcnn_class_conv2")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_class_bn2')(x, training=train_bn)
        shared = keras.layers.Activation('relu')(x)
        shared = keras.layers.Reshape((rois_size,fc_layers_size))(shared)
        logits = keras.layers.Dense(num_classes, name='mrcnn_class_logits')(shared)
        probabilities = keras.layers.Activation("softmax", name="mrcnn_class")(logits)
        bounding_boxes = keras.layers.Dense(num_classes * 4, activation='linear', name='mrcnn_bbox_fc')(shared)
        bounding_boxes = keras.layers.Reshape((rois_size,num_classes,4))(bounding_boxes)
        return probabilities,bounding_boxes
    
class FPNMaskGraph():

    def __init__(self, rois, feature_maps,pool_size, num_classes, train_bn=False):
        self.rois = rois
        self.feature_maps = feature_maps
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.train_bn = train_bn

    def _build_inner_model(self):
        num_classes = self.num_classes
        train_bn = self.train_bn
        # Conv layers
        input = keras.layers.Input((14,14,256))
        x = keras.layers.Conv2D(256, (3, 3), padding="same",name = "mrcnn_mask_conv1")(input)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn1')(x, training=train_bn)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same",name = "mrcnn_mask_conv2")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn2')(x, training=train_bn)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same",name = "mrcnn_mask_conv3")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn3')(x, training=train_bn)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same",name = "mrcnn_mask_conv4")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn4')(x, training=train_bn)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu",name = "mrcnn_mask_deconv")(x)
        x = keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid",name = "mrcnn_mask")(x)
        x = keras.layers.Reshape((28,28,num_classes))(x)
        return keras.models.Model(inputs=[input], outputs=[x])

    def build(self):
        pool_size = self.pool_size
        rois = self.rois
        feature_maps = self.feature_maps
        num_classes= self.num_classes

        COCO_MODEL_PATH = "Data/weights.h5"
        fpn_mask_model = self._build_inner_model()
        fpn_mask_model.load_weights(COCO_MODEL_PATH, by_name=True)

        pyramid = PyramidROIAlign([pool_size, pool_size],
                            name="roi_align_mask")([rois] + feature_maps)

        x = FixedTimeDistributed(fpn_mask_model, 100, ["mrcnn_mask_concat"],[(784,num_classes)])(pyramid)
        result= keras.layers.Reshape((28,28,100))(x)

        fpn_mask_model = self._build_inner_model()
        fpn_mask_model.load_weights(COCO_MODEL_PATH, by_name=True)
        return fpn_mask_model, result