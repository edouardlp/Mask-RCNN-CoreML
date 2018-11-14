import coremltools
import tensorflow as tf

model = tf.keras.models.load_model("output/coco/keras_model.h5")

coremltools.converters.keras.convert(model)