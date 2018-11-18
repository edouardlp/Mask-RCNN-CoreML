import os
import math
import json
import keras
import coremltools
import skimage.io
import numpy as np

from coremltools.proto import NeuralNetwork_pb2

import subgraphs.resnet101_fpn_backbone_graph as resnet101_fpn_backbone_graph
import subgraphs.rpn_graph as rpn_graph
import subgraphs.proposal_layer as proposal_layer
import subgraphs.fpn_classifier_graph as fpn_classifier_graph
import subgraphs.fpn_mask_graph as fpn_mask_graph
import subgraphs.detection_layer as detection_layer

def build_models(config_path,
                 weights_path,
                 anchors_path):

    input_width = 1024
    input_height = 1024
    num_classes = 81
    max_proposals = 1000
    max_detections = 100
    pyramid_top_down_size = 256
    anchor_stride = 1
    anchors_per_location = len([0.5, 1, 2])
    proposal_nms_threshold = 0.7

    input_image = keras.layers.Input(shape=[input_width,input_height,3], name="input_image")

    backbone = resnet101_fpn_backbone_graph.BackboneGraph(input_tensor=input_image,
                                                       architecture = 'resnet101',
                                                       pyramid_size = pyramid_top_down_size)
    P2, P3, P4, P5, P6 = backbone.build()

    rpn_class, rpn_bbox = rpn_graph.RPNGraph(anchor_stride=anchor_stride,
                                             anchors_per_location=anchors_per_location,
                                             depth=pyramid_top_down_size,
                                             feature_maps=[P2, P3, P4, P5, P6]).build()

    rpn_rois = proposal_layer.ProposalLayer(max_proposals=max_proposals,
                                            nms_threshold=proposal_nms_threshold,
                                            name="ROI")([rpn_class, rpn_bbox])


    mrcnn_feature_maps = [P2, P3, P4, P5]
    classifier_pool_size = 7
    mask_pool_size = 14

    #TODO:fc_layers_size
    mrcnn_class, mrcnn_bbox = fpn_classifier_graph.FPNClassifierGraph(rois=rpn_rois,
                                                                      feature_maps=mrcnn_feature_maps,
                                                                      pool_size=classifier_pool_size,
                                                                      num_classes=num_classes,
                                                                      max_regions=max_proposals,
                                                                      fc_layers_size=1024,
                                                                      pyramid_top_down_size=pyramid_top_down_size).build()

    detections = detection_layer.DetectionLayer(name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox])
    #TODO: try to remove this line
    detections = keras.layers.Reshape((max_detections,6))(detections)

    fpn_mask_model, mrcnn_mask = fpn_mask_graph.FPNMaskGraph(rois=detections,
                                                             feature_maps=mrcnn_feature_maps,
                                                             pool_size=mask_pool_size,
                                                             num_classes=num_classes,
                                                             max_regions=max_detections,
                                                             pyramid_top_down_size=pyramid_top_down_size,
                                                             weights_path=weights_path).build()

    mask_rcnn_model = keras.models.Model(input_image,
                                     [detections, mrcnn_mask],
                                     name='mask_rcnn_model')

    mask_rcnn_model.load_weights(weights_path, by_name=True)
    return mask_rcnn_model, fpn_mask_model

def export_models(mask_rcnn_model,
                  mask_model,
                  export_main_path,
                  export_mask_path,
                  export_anchors_path):
    
    license = "MIT"
    author = "Édouard Lavery-Plante"

    def convert_proposal(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "ProposalLayer"
        params.description = ""
        #boundingBoxRefinementStandardDeviation
        #preNonMaxSupressionLimit
        #proposalLimit
        #nonMaxSupressionInteresectionOverUnionThreshold
        return params

    def convert_pyramid(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "PyramidROIAlignLayer"
        params.parameters["poolSize"].intValue = layer.pool_shape[0]
        # imageSize = CGSize(width: 1024, height: 1024)
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

    mask_rcnn_model = coremltools.converters.keras.convert(mask_rcnn_model,
                                                           input_names=["image"],
                                                           image_input_names=['image'],
                                                           output_names=["detections", "mask"],
                                                           add_custom_layers=True,
                                                           custom_conversion_functions={"ProposalLayer": convert_proposal,
                                                                                        "PyramidROIAlign": convert_pyramid,
                                                                                        "TimeDistributedMask": convert_time_distributed,
                                                                                        "DetectionLayer": convert_detection})

    mask_rcnn_model.author = author
    mask_rcnn_model.license = license
    mask_rcnn_model.short_description = "Mask-RCNN"
    mask_rcnn_model.input_description["image"] = "Input image"
    full_spec = mask_rcnn_model.get_spec()
    half_model = coremltools.models.utils.convert_neural_network_weights_to_fp16(mask_rcnn_model)
    half_spec = half_model.get_spec()
    coremltools.utils.save_spec(full_spec, export_main_path)
    
    mask_model_coreml = coremltools.converters.keras.convert(mask_model,
                                                             input_names=["pooled_region"],
                                                             output_names=["mask"])
    mask_model_coreml.author = author
    mask_model_coreml.license = license
    mask_model_coreml.short_description = "Mask"
    half_mask_model = coremltools.models.utils.convert_neural_network_weights_to_fp16(mask_model_coreml)
    half_mask_spec = half_mask_model.get_spec()
    coremltools.utils.save_spec(half_mask_spec, export_mask_path)


def export(config_path,
           weights_path,
           anchors_path,
           export_main_path,
           export_mask_path,
           export_anchors_path,
           params):

    mask_rcnn_model, mask_model = build_models(config_path,weights_path,anchors_path)
    export_models(mask_rcnn_model, mask_model, export_main_path, export_mask_path, export_anchors_path)
    anchors = np.load(anchors_path)
    anchors.tofile(export_anchors_path)
