import os
import math
import json
import keras
import coremltools
import skimage.io
import numpy as np
from keras.utils import plot_model
from coremltools.proto import NeuralNetwork_pb2

import subgraphs.resnet101_fpn_backbone_graph as resnet101_fpn_backbone_graph
import subgraphs.rpn_graph as rpn_graph
import subgraphs.proposal_layer as proposal_layer
import subgraphs.fpn_classifier_graph as fpn_classifier_graph
import subgraphs.detection_layer as detection_layer

def export_models(mask_rcnn_model, fpn_classifier_model):

    def convert_proposal(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "ProposalLayer"
        params.description = ""
        return params

    def convert_pyramid(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "PyramidROIAlignLayer"
        params.parameters["poolSize"].intValue = 7
        params.description = ""
        return params

    def convert_time_distributed(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "FixedTimeDistributedLayer"
        params.description = ""
        return params

    def convert_detection(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "DetectionLayer"
        params.description = ""
        return params

    def convert_debug(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "DebugLayer"
        params.description = ""
        return params


    mask_rcnn_model = coremltools.converters.keras.convert(mask_rcnn_model,
                                                           input_names=["image"],
                                                           image_input_names=['image'],
                                                           output_names=["detections", "mask"],
                                                           add_custom_layers=True,
                                                           custom_conversion_functions={"ProposalLayer": convert_proposal,
                                                                                        "PyramidROIAlign": convert_pyramid,
                                                                                        "FixedTimeDistributed": convert_time_distributed,
                                                                                        "DetectionLayer": convert_detection,
                                                                                        "DebugLayer": convert_debug})

    mask_rcnn_model.author = "Édouard Lavery-Plante"
    mask_rcnn_model.license = "MIT"
    mask_rcnn_model.short_description = "Mask-RCNN"
    mask_rcnn_model.input_description["image"] = "Input image"
    mask_rcnn_model.save("MaskRCNNCrop.mlmodel")

    if(fpn_classifier_model):
        fpn_classifier_coreml = coremltools.converters.keras.convert(fpn_classifier_model,
                                                                 input_names=["pooled_region"],
                                                                 output_names=["mask"])
        fpn_classifier_coreml.author = "Édouard Lavery-Plante"
        fpn_classifier_coreml.license = "MIT"
        fpn_classifier_coreml.short_description = "FPN-Massk"
        fpn_classifier_coreml.save("FPNMask.mlmodel")

if __name__ == '__main__':

    anchors = np.load("anchors.npy")
    anchors.tofile("anchors.bin")

    COCO_MODEL_PATH = "mask_rcnn_coco.h5"
    input_image = keras.layers.Input(shape=[1024,1024,3], name="input_image")

    graph = resnet101_fpn_backbone_graph.BackboneGraph(input_tensor=input_image)
    P2, P3, P4, P5, P6 = graph.build()

    rpn_feature_maps = [P2, P3, P4, P5, P6]
    mrcnn_feature_maps = [P2, P3, P4, P5]

    rpn = rpn_graph.RPNGraph().build()

    layer_outputs = []
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))
    output_names = ["rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [keras.layers.Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]

    rpn_class, rpn_bbox = outputs

    proposal_count = 1000
    rpn_rois = proposal_layer.ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=0.7,
        name="ROI")([rpn_class, rpn_bbox])

    mrcnn_class, mrcnn_bbox = \
        fpn_classifier_graph.FPNClassifierGraph(rpn_rois, mrcnn_feature_maps).build()

    mrcnn_bbox = keras.layers.Permute((3,2,1))(mrcnn_bbox)

    detections = detection_layer.DetectionLayer(name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox])
    detections = keras.layers.Reshape((100,6))(detections)

    fpn_mask_model, mrcnn_mask = fpn_classifier_graph.FPNMaskGraph(detections, mrcnn_feature_maps,14,81).build()

    mask_rcnn_model = keras.models.Model(input_image,
                                   [detections],
                                   name='mask_rcnn_model')

    mask_rcnn_model.load_weights(COCO_MODEL_PATH, by_name=True)
    print(mask_rcnn_model.summary())

    image = skimage.io.imread("/Users/elaveryplante/Desktop/IMG_0162.jpeg")
    cnn_image = np.reshape(image, (1, 1024, 1024, 3))

    results = mask_rcnn_model.predict([cnn_image])
    print(results[0])
    #np.save("p5.npy", results[0])
    #print(results[0].shape)
    #print(results[1].shape)

    #export_models(mask_rcnn_model, fpn_mask_model)

