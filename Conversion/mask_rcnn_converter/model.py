import os
import math
import json
import keras
import coremltools
import skimage.io
import numpy as np

from coremltools.proto import NeuralNetwork_pb2

from subgraphs.fpn_backbone_graph import BackboneGraph
from subgraphs.rpn_graph import RPNGraph
from subgraphs.proposal_layer import ProposalLayer
from subgraphs.fpn_classifier_graph import FPNClassifierGraph
from subgraphs.fpn_mask_graph import FPNMaskGraph
from subgraphs.detection_layer import DetectionLayer

def build_models(config_path,
                 weights_path):

    #TODO: load from config_path

    architecture = 'resnet101'
    input_width = 1024
    input_height = 1024
    input_image_shape = (input_width,input_height)
    num_classes = 1+80
    pre_nms_max_proposals = 6000
    max_proposals = 1000
    max_detections = 100
    pyramid_top_down_size = 256
    proposal_nms_threshold = 0.7
    detection_min_confidence = 0.7
    detection_nms_threshold = 0.3
    bounding_box_std_dev = [0.1, 0.1, 0.2, 0.2]
    classifier_pool_size = 7
    mask_pool_size = 14
    fc_layers_size = 1024
    anchor_scales = (32, 64, 128, 256, 512)
    anchor_ratios = [0.5, 1, 2]
    anchors_per_location = len(anchor_ratios)
    backbone_strides = [4, 8, 16, 32, 64]
    anchor_stride = 1

    input_image = keras.layers.Input(shape=[input_width,input_height,3], name="input_image")

    backbone = BackboneGraph(input_tensor=input_image,
                             architecture = architecture,
                             pyramid_size = pyramid_top_down_size)

    P2, P3, P4, P5, P6 = backbone.build()

    rpn = RPNGraph(anchor_stride=anchor_stride,
                   anchors_per_location=anchors_per_location,
                   depth=pyramid_top_down_size,
                   feature_maps=[P2, P3, P4, P5, P6])

    #anchor_object_probs: Probability of each anchor containing only background or objects
    #anchor_deltas: Bounding box refinements to apply to each anchor to better enclose its object
    anchor_object_probs, anchor_deltas = rpn.build()

    #rois: Regions of interest (regions of the image that probably contain an object)
    proposal_layer = ProposalLayer(name="ROI",
                                   image_shape=input_image_shape,
                                   max_proposals=max_proposals,
                                   pre_nms_max_proposals=pre_nms_max_proposals,
                                   bounding_box_std_dev=bounding_box_std_dev,
                                   nms_threshold=proposal_nms_threshold,
                                   anchor_scales=anchor_scales,
                                   anchor_ratios=anchor_ratios,
                                   backbone_strides=backbone_strides,
                                   anchor_stride=anchor_stride)

    rois = proposal_layer([anchor_object_probs, anchor_deltas])

    mrcnn_feature_maps = [P2, P3, P4, P5]

    fpn_classifier_graph = FPNClassifierGraph(rois=rois,
                                              feature_maps=mrcnn_feature_maps,
                                              pool_size=classifier_pool_size,
                                              image_shape=input_image_shape,
                                              num_classes=num_classes,
                                              max_regions=max_proposals,
                                              fc_layers_size=fc_layers_size,
                                              pyramid_top_down_size=pyramid_top_down_size,
                                              weights_path=weights_path)

    #rois_class_probs: Probability of each class being contained within the roi
    #rois_deltas: Bounding box refinements to apply to each roi to better enclose its object
    fpn_classifier_model, classification = fpn_classifier_graph.build()


    detections = DetectionLayer(name="mrcnn_detection",
                                max_detections=max_detections,
                                bounding_box_std_dev=bounding_box_std_dev,
                                detection_min_confidence=detection_min_confidence,
                                detection_nms_threshold=detection_nms_threshold)([rois, classification])

    #TODO: try to remove this line
    detections = keras.layers.Reshape((max_detections,6))(detections)

    fpn_mask_graph = FPNMaskGraph(rois=detections,
                                  feature_maps=mrcnn_feature_maps,
                                  pool_size=mask_pool_size,
                                  image_shape=input_image_shape,
                                  num_classes=num_classes,
                                  max_regions=max_detections,
                                  pyramid_top_down_size=pyramid_top_down_size,
                                  weights_path=weights_path)

    fpn_mask_model, masks = fpn_mask_graph.build()

    mask_rcnn_model = keras.models.Model(input_image,
                                     [detections, masks],
                                     name='mask_rcnn_model')

    mask_rcnn_model.load_weights(weights_path, by_name=True)
    return mask_rcnn_model, fpn_classifier_model, fpn_mask_model, proposal_layer.anchors

def export_models(mask_rcnn_model,
                  classifier_model,
                  mask_model,
                  export_main_path,
                  export_mask_path,
                  export_anchors_path):
    
    license = "MIT"
    author = "Édouard Lavery-Plante"

    def convert_proposal(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "ProposalLayer"
        params.description = "Proposes regions of interests and performs NMS."
        params.parameters["bboxStdDev_count"].intValue = len(layer.bounding_box_std_dev)
        for idx,value in enumerate(layer.bounding_box_std_dev):
            params.parameters["bboxStdDev_"+str(idx)].doubleValue = value
        params.parameters["preNMSMaxProposals"].intValue = layer.pre_nms_max_proposals
        params.parameters["maxProposals"].intValue = layer.max_proposals
        params.parameters["nmsIOUThreshold"].doubleValue = layer.nms_threshold
        return params

    def convert_pyramid(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "PyramidROIAlignLayer"
        params.parameters["poolSize"].intValue = layer.pool_shape[0]
        params.parameters["imageWidth"].intValue = layer.image_shape[0]
        params.parameters["imageHeight"].intValue = layer.image_shape[1]
        params.description = "Extracts feature maps based on the regions of interest."
        return params

    def convert_time_distributed_classifier(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "TimeDistributedClassifierLayer"
        return params

    def convert_time_distributed(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "TimeDistributedMaskLayer"
        params.description = "Applies the Mask graph to each detections along the time dimension."
        return params

    def convert_detection(layer):
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = "DetectionLayer"
        params.parameters["bboxStdDev_count"].intValue = len(layer.bounding_box_std_dev)
        for idx,value in enumerate(layer.bounding_box_std_dev):
            params.parameters["bboxStdDev_"+str(idx)].doubleValue = value
        params.parameters["maxDetections"].intValue = layer.max_detections
        params.parameters["scoreThreshold"].doubleValue = layer.detection_min_confidence
        params.parameters["nmsIOUThreshold"].doubleValue = layer.detection_nms_threshold
        params.description = "Outputs detections based on confidence and performs NMS."
        return params

    mask_rcnn_model = coremltools.converters.keras.convert(mask_rcnn_model,
                                                           input_names=["image"],
                                                           image_input_names=['image'],
                                                           output_names=["detections", "mask"],
                                                           add_custom_layers=True,
                                                           custom_conversion_functions={"ProposalLayer": convert_proposal,
                                                                                        "PyramidROIAlign": convert_pyramid,
                                                                                        "TimeDistributedClassifier" : convert_time_distributed_classifier,
                                                                                        "TimeDistributedMask": convert_time_distributed,
                                                                                        "DetectionLayer": convert_detection})

    mask_rcnn_model.author = author
    mask_rcnn_model.license = license
    mask_rcnn_model.short_description = "Mask-RCNN"
    mask_rcnn_model.input_description["image"] = "Input image"
    mask_rcnn_model.output_description["detections"] = "Detections (y1,x1,y2,x2,classId,score)"
    mask_rcnn_model.output_description["mask"] = "Masks for the detections"
    half_model = coremltools.models.utils.convert_neural_network_weights_to_fp16(mask_rcnn_model)
    half_spec = half_model.get_spec()
    coremltools.utils.save_spec(half_spec, export_main_path)
    
    mask_model_coreml = coremltools.converters.keras.convert(mask_model,
                                                             input_names=["feature_map"],
                                                             output_names=["masks"])
    mask_model_coreml.author = author
    mask_model_coreml.license = license
    mask_model_coreml.short_description = "Generates a mask for each class for a given feature map"
    mask_model_coreml.input_description["feature_map"] = "Fully processed feature map, ready for mask generation."
    mask_model_coreml.output_description["masks"] = "Masks corresponding to each class"
    half_mask_model = coremltools.models.utils.convert_neural_network_weights_to_fp16(mask_model_coreml)
    half_mask_spec = half_mask_model.get_spec()
    coremltools.utils.save_spec(half_mask_spec, export_mask_path)

    classifier_model_coreml = coremltools.converters.keras.convert(classifier_model,
                                                             input_names=["feature_map"],
                                                             output_names=["probabilities", "bounding_boxes"])
    classifier_model_coreml.author = author
    classifier_model_coreml.license = license
    classifier_model_coreml.short_description = ""
    classifier_model_coreml.input_description["feature_map"] = "Fully processed feature map, ready for classification."
    half_classifier_model = coremltools.models.utils.convert_neural_network_weights_to_fp16(classifier_model_coreml)
    half_classifier_spec = half_classifier_model.get_spec()
    coremltools.utils.save_spec(half_classifier_spec, "Data/Classifier.mlmodel")


def export(config_path,
           weights_path,
           export_main_path,
           export_mask_path,
           export_anchors_path,
           params):

    mask_rcnn_model, classifier_model, mask_model,anchors = build_models(config_path,weights_path)
    export_models(mask_rcnn_model, classifier_model, mask_model, export_main_path, export_mask_path, export_anchors_path)
    anchors.tofile(export_anchors_path)
