//
//  ProposalLayer.swift
//  Mask-RCNN-CoreML
//
//  Created by Edouard Lavery-Plante on 2018-10-26.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

/**
 
 ProposalLayer is a Custom ML Layer that proposes regions of interests.
 
 ProposalLayer proposes regions of interest based on the probability of objects
 being detected in each region. Regions that overlap more than a given
 threshold are removed through a process called Non-Max Supression (NMS).
 
 Regions correspond to predefined "anchors" that are not inputs to the layer.
 Anchors are generated based on the image shape using a heuristic that maximizes
 the likelihood of bounding objects in the image. The process of generating anchors
 can be though of as a hyperparameter.
 
 Anchors are adjusted using deltas provided as input to refine how they enclose the
 detected objects.

 The layer takes two inputs :
 
 - Probabilities of each region containing an object. Shape : (#regions, 2).
 - Anchor deltas to refine the anchors shape. Shape : (#regions,4)
 
 The probabilities input's last dimension corresponds to the mutually exclusive
 probabilities of the region being background (index 0) or an object (index 1).
 
 The anchor deltas are layed out as follows : (dy,dx,log(dh),log(dw)).
 
 The layer takes four parameters :
 
 - boundingBoxRefinementStandardDeviation : Anchor deltas refinement standard deviation
 - preNMSMaxProposals : Maximum # of regions to evaluate for non max supression
 - maxProposals : Maximum # of regions to output
 - nmsIOUThreshold : Threshold below which to supress regions that overlap
 
 The layer has one ouput :
 
 - Regions of interest (y1,x1,y2,x2). Shape : (#regionsOut,4),
 
 */
@objc(ProposalLayer) class ProposalLayer: NSObject, MLCustomLayer {
    
    var anchorData:Data!
    
    //Anchor deltas refinement standard deviation
    var boundingBoxRefinementStandardDeviation:[Float] = [0.1, 0.1, 0.2, 0.2]
    //Maximum # of regions to evaluate for non max supression
    var preNMSMaxProposals = 6000
    //Maximum # of regions to output
    var maxProposals = 1000
    //Threshold below which to supress regions that overlap
    var nmsIOUThreshold:Float = 0.7
    
    required init(parameters: [String : Any]) throws {
        super.init()
        //TODO: generate the anchors on demand based on image shape
        self.anchorData = try Data(contentsOf: Bundle.main.url(forResource: "anchors", withExtension: "bin")!)
        
        if let bboxStdDevCount = parameters["bboxStdDev_count"] as? Int {
            var bboxStdDev = [Float]()
            for i in 0..<bboxStdDevCount {
                if let bboxStdDevItem = parameters["bboxStdDev_\(i)"] as? Double {
                    bboxStdDev.append(Float(bboxStdDevItem))
                }
            }
            if(bboxStdDev.count == bboxStdDevCount){
                self.boundingBoxRefinementStandardDeviation = bboxStdDev
            }
        }
        
        if let preNMSMaxProposals = parameters["preNMSMaxProposals"] as? Int {
            self.preNMSMaxProposals = preNMSMaxProposals
        }
        if let maxProposals = parameters["maxProposals"] as? Int {
            self.maxProposals = maxProposals
        }
        if let nmsIOUThreshold = parameters["nmsIOUThreshold"] as? Double {
            self.nmsIOUThreshold = Float(nmsIOUThreshold)
        }
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        var outputshape = inputShapes[1]
        outputshape[0] = NSNumber(integerLiteral: self.maxProposals)
        return [outputshape]
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)
        assert(inputs[1].dataType == MLMultiArrayDataType.float32)

        //Probabilities of each region containing an object. Shape : (#regions, 2).
        let classProbabilities = inputs[0]
        //Anchor deltas to refine the anchors shape. Shape : (#regions,4)
        let anchorDeltas = inputs[1]

        let preNonMaxLimit = self.preNMSMaxProposals
        let maxProposals = self.maxProposals
        
        let totalNumberOfElements = Int(truncating:classProbabilities.shape[0])
        let numberOfElementsToProcess = min(totalNumberOfElements, preNonMaxLimit)
        
        //We extract only the object probabilities, which are always at the odd indices of the array
        let objectProbabilities = classProbabilities.floatDataPointer().stridedSlice(begin: 1, count: totalNumberOfElements, stride: 2)
        
        //We sort the probabilities in descending order and get the index so as to reorder the other arrays.
        //We also clip to the limit.
        let sortedProbabilityIndices = objectProbabilities.sortedIndices(ascending: false)[0 ..< numberOfElementsToProcess].toFloat()
        
        //We broadcast the probability indices so that they index the boxes (anchor deltas and anchors)
        let boxElementLength = 4
        let boxIndices = broadcastedIndices(indices: sortedProbabilityIndices, toElementLength: boxElementLength)
        
        //We sort the deltas and the anchors
        
        var sortedDeltas:BoxArray = anchorDeltas.floatDataPointer().indexed(indices: boxIndices)
        
        var sortedAnchors:BoxArray = self.anchorData.withUnsafeBytes {
            (data:UnsafePointer<Float>) -> [Float] in
            return data.indexed(indices: boxIndices)
        }
        
        //For each element of deltas, multiply by stdev
        
        var stdDev = self.boundingBoxRefinementStandardDeviation
        let stdDevPointer = UnsafeMutablePointer<Float>(&stdDev)
        elementWiseMultiply(matrixPointer: UnsafeMutablePointer<Float>(&sortedDeltas), vectorPointer: stdDevPointer, height:numberOfElementsToProcess, width: stdDev.count)
        
        //We apply the box deltas and clip the results in place to the image boundaries
        let anchorsReference = sortedAnchors.boxReference()
        anchorsReference.applyBoxDeltas(sortedDeltas)
        anchorsReference.clip()
        
        //We apply Non Max Supression to the result boxes
        
        let resultIndices = nonMaxSupression(boxes: sortedAnchors,
                                             indices: Array(0 ..< sortedAnchors.count),
                                             iouThreshold: self.nmsIOUThreshold,
                                             max: maxProposals)
        
        //We copy the result boxes corresponding to the resultIndices to the output
        
        let output = outputs[0]
        let outputElementStride = Int(truncating: output.strides[0])
        
        for (i,resultIndex) in resultIndices.enumerated() {
            for j in 0 ..< 4 {
                output[i*outputElementStride+j] = sortedAnchors[resultIndex*4+j] as NSNumber
            }
        }
        
        //Zero-pad the rest since CoreML does not erase the memory between evaluations

        let proposalCount = resultIndices.count
        let paddingCount = max(0,maxProposals-proposalCount)*outputElementStride
        output.padTailWithZeros(startIndex: proposalCount*outputElementStride, count: paddingCount)
    }
    
}

//
//def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
//"""
//scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
//ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
//shape: [height, width] spatial shape of the feature map over which
//to generate anchors.
//feature_stride: Stride of the feature map relative to the image in pixels.
//anchor_stride: Stride of anchors on the feature map. For example, if the
//value is 2 then generate anchors for every other feature map pixel.
//"""
//# Get all combinations of scales and ratios
//scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
//scales = scales.flatten()
//ratios = ratios.flatten()
//
//# Enumerate heights and widths from scales and ratios
//heights = scales / np.sqrt(ratios)
//widths = scales * np.sqrt(ratios)
//
//# Enumerate shifts in feature space
//shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
//shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
//shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
//
//# Enumerate combinations of shifts, widths, and heights
//box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
//box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
//
//# Reshape to get a list of (y, x) and a list of (h, w)
//box_centers = np.stack(
//[box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
//box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
//
//# Convert to corner coordinates (y1, x1, y2, x2)
//boxes = np.concatenate([box_centers - 0.5 * box_sizes,
//box_centers + 0.5 * box_sizes], axis=1)
//return boxes
//
//
//def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
//                             anchor_stride):
//"""Generate anchors at different levels of a feature pyramid. Each scale
//is associated with a level of the pyramid, but each ratio is used in
//all levels of the pyramid.
//
//Returns:
//anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
//with the same order of the given scales. So, anchors of scale[0] come
//first, then anchors of scale[1], and so on.
//"""
//# Anchors
//# [anchor_count, (y1, x1, y2, x2)]
//anchors = []
//for i in range(len(scales)):
//anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
//feature_strides[i], anchor_stride))
//return np.concatenate(anchors, axis=0)
