//
//  ProposalLayer.swift
//  Mask-RCNN-Demo
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
 
 - Probabilities of a region containing an object. Shape : (#regions, 2).
 - Anchor deltas to refine the anchors shape. Shape : (#regions,4)
 
 The probabilities input's last dimension corresponds to the mutually exclusive
 probabilities of the region being background (index 0) or an object (index 1).
 
 The anchor deltas are layed out as follows : (dy,dx,log(dh),log(dw)).
 
 The layer takes three parameters :
 
 - boundingBoxRefinementStandardDeviation : Anchor deltas refinement standard deviation
 - preNonMaxSupressionLimit : Maximum # of regions to evaluate for non max supression
 - proposalLimit : Maximum # of regions to output
 
 The layer has one ouput :
 
 - Regions of interest. Shape : (#regionsOut,4),
 
 */
@objc(ProposalLayer) class ProposalLayer: NSObject, MLCustomLayer {
    
    var anchorData:Data!
    
    //Anchor deltas refinement standard deviation
    let boundingBoxRefinementStandardDeviation:[Float] = [0.1, 0.1, 0.2, 0.2]
    //Maximum # of regions to evaluate for non max supression
    var preNonMaxSupressionLimit = 6000
    //Maximum # of regions to output
    var proposalLimit = 1000
    
    required init(parameters: [String : Any]) throws {
        super.init()
        self.anchorData = try Data(contentsOf: Bundle.main.url(forResource: "anchors", withExtension: "bin")!)
        //TODO: load stdev and limits
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        var outputshape = inputShapes[1]
        outputshape[0] = NSNumber(integerLiteral: self.proposalLimit)
        return [outputshape]
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)
        
        let log = OSLog(subsystem: "Proposal", category: OSLog.Category.pointsOfInterest)
        os_signpost(.begin, log: log, name: "Proposal-Eval")
        
        let classProbabilities = inputs[0]
        let anchorDeltas = inputs[1]

        let preNonMaxLimit = self.preNonMaxSupressionLimit
        let maxProposals = self.proposalLimit
        
        let foregroundProbability = extractForegroundProbabilities(rpnClassMultiArray: classProbabilities)
        var (sortedProbabilities,sortedProbabilitiesIndices) = sortProbabilities(probabilities: foregroundProbability, keepTop: preNonMaxLimit)
        
        let numberOfElementsToProcess = vDSP_Length(sortedProbabilities.count)
        
        let boxElementLength:UInt = 4
        let boxCount = numberOfElementsToProcess*boxElementLength
        
        var boxIndices:[Float] = Array(repeating: 0, count: Int(boxCount))
        let sortedProbabilitiesIndicesPointer = UnsafeMutablePointer<Float>(&sortedProbabilitiesIndices)
        let boxIndicesPointer = UnsafeMutablePointer<Float>(&boxIndices)
        computeIndices(fromIndicesPointer: sortedProbabilitiesIndicesPointer, toIndicesPointer: boxIndicesPointer, elementLength: boxElementLength, elementCount: numberOfElementsToProcess)
        
        let deltasPointer = UnsafeMutablePointer<Float>(OpaquePointer(anchorDeltas.dataPointer))
        var sortedDeltas:[Float] = Array<Float>(repeating: 0, count: Int(boxCount))
        let sortedDeltasPointer = UnsafeMutablePointer<Float>(&sortedDeltas)
        vDSP_vindex(deltasPointer, boxIndicesPointer, 1, sortedDeltasPointer, 1, vDSP_Length(boxCount))
        
        var sortedAnchors:[Float] = Array<Float>(repeating: 0, count: Int(boxCount))
        let sortedAnchorPointer = UnsafeMutablePointer<Float>(&sortedAnchors)
        
        self.anchorData.withUnsafeBytes {
            (data:UnsafePointer<Float>) in
            vDSP_vindex(data, boxIndicesPointer, 1, sortedAnchorPointer, 1, vDSP_Length(boxCount))
        }
        
        //For each element of deltas, multiply by stdev
        
        var stdDev = self.boundingBoxRefinementStandardDeviation
        let stdDevPointer = UnsafeMutablePointer<Float>(&stdDev)
        elementWiseMultiply(matrixPointer: sortedDeltasPointer, vectorPointer: stdDevPointer, height:Int(numberOfElementsToProcess), width: stdDev.count)
        var resultBoxes = applyBoxDeltas(boxes: sortedAnchors, deltas: sortedDeltas)
        let resultBoxesPointer = UnsafeMutablePointer<Float>(&resultBoxes)
        clipBoxes(boxesPointer: resultBoxesPointer, elementCount: Int(numberOfElementsToProcess))
        
        let resultIndices = nonMaxSupression(boxes: resultBoxes,
                                             indices: Array(0 ..< resultBoxes.count),
                                             iouThreshold: 0.7,
                                             max: maxProposals)
        
        let output = outputs[0]
        let outputElementStride = Int(truncating: output.strides[0])
        for (i,resultIndex) in resultIndices.enumerated() {
            for j in 0 ..< 4 {
                output[i*outputElementStride+j] = resultBoxes[resultIndex*4+j] as NSNumber
            }
        }
        
        let proposalCount = resultIndices.count
        let paddingCount = max(0,maxProposals-proposalCount)*outputElementStride
        output.padTailWithZeros(startIndex: proposalCount*outputElementStride, count: paddingCount)
        
        os_signpost(.end, log: log, name: "Proposal-Eval")
    }
    
}

func extractForegroundProbabilities(rpnClassMultiArray:MLMultiArray) -> [Float] {
    
    let totalElements = Int(truncating:rpnClassMultiArray.shape[0])
    //The successor points to the foreground class of the first element
    let rpnClassPointer = UnsafeMutablePointer<Float>(OpaquePointer(rpnClassMultiArray.dataPointer)).successor()
    
    var foregroundProbability:[Float] = Array(repeating: 0, count: totalElements)
    let foregroundProbabilityPointer = UnsafeMutablePointer<Float>(&foregroundProbability)
    cblas_scopy(Int32(totalElements), rpnClassPointer, 2, foregroundProbabilityPointer, 1)
    return foregroundProbability
}

func sortProbabilities(probabilities:[Float], keepTop:Int) -> ([Float],[Float]) {
    
    var probabilities = probabilities
    let probabilitiesPointer = UnsafeMutablePointer<Float>(&probabilities)
    let totalElements = vDSP_Length(probabilities.count)
    
    var indices = Array<vDSP_Length>(0 ..< totalElements)
    let indicesPointer = UnsafeMutablePointer<vDSP_Length>(&indices)
    vDSP_vsorti(probabilitiesPointer, indicesPointer, nil, totalElements, -1)
    
    //We clip to the limit
    let numberOfElementsToProcess = min(totalElements, vDSP_Length(keepTop))
    
    var floatIndices:[Float] = Array<Float>(indices[0..<Int(numberOfElementsToProcess)].map({ (integerValue) -> Float in
        return Float(integerValue)
    }))
    let floatIndicesPointer = UnsafeMutablePointer<Float>(&floatIndices)
    
    var sortedForegroundProbability:[Float] = Array<Float>(repeating: 0, count: Int(numberOfElementsToProcess))
    let sortedForegroundProbabilityPointer = UnsafeMutablePointer<Float>(&sortedForegroundProbability)
    
    vDSP_vindex(probabilitiesPointer, floatIndicesPointer, 1, sortedForegroundProbabilityPointer, 1, numberOfElementsToProcess)
    
    return (sortedForegroundProbability,floatIndices)
}

func computeIndices(fromIndicesPointer:UnsafeMutablePointer<Float>,
                    toIndicesPointer:UnsafeMutablePointer<Float>,
                    elementLength:UInt,
                    elementCount:UInt) {
    cblas_scopy(Int32(elementCount), fromIndicesPointer, 1, toIndicesPointer, Int32(elementLength))
    
    for i in 1 ..< Int(elementLength) {
        cblas_scopy(Int32(elementCount), fromIndicesPointer, 1, toIndicesPointer.advanced(by: i), Int32(elementLength))
    }
    
    var multiplicationScalar:Float = Float(elementLength)
    let multiplicationScalarPointer = UnsafeMutablePointer<Float>(&multiplicationScalar)
    vDSP_vsmul(toIndicesPointer, 1, multiplicationScalarPointer, toIndicesPointer, 1, elementLength*elementCount)
    
    for i in 1 ..< Int(elementLength) {
        
        var shift:Float = Float(i)
        let shiftPointer = UnsafeMutablePointer<Float>(&shift)
        
        vDSP_vsadd(toIndicesPointer.advanced(by: i), vDSP_Stride(elementLength), shiftPointer, toIndicesPointer.advanced(by: i), vDSP_Stride(elementLength), elementCount)
    }
}

