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

@objc(ProposalLayer) class ProposalLayer: NSObject, MLCustomLayer {
    
    let preNonMaxLimit:UInt = 6000
    var anchorData:Data!
    // Bounding box refinement standard deviation
    let boundingBoxRefinementStandardDeviation:[Float] = [0.1, 0.1, 0.2, 0.2]
    
    required init(parameters: [String : Any]) throws {
        super.init()
        self.anchorData = try Data(contentsOf: Bundle.main.url(forResource: "anchors", withExtension: "bin")!)
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        var outputshape = inputShapes[1]
        outputshape[0] = NSNumber(integerLiteral: 1000)
        return [outputshape]//1000,1,4,1,1
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)
        let log = OSLog(subsystem: "Proposal", category: OSLog.Category.pointsOfInterest)
        os_signpost(.begin, log: log, name: "Proposal-Eval")

        let preNonMaxLimit = self.preNonMaxLimit
        
        let rpnClass = inputs[0]
        let anchorDeltas = inputs[1]
        
        let foregroundProbability = extractForegroundProbabilities(rpnClassMultiArray: rpnClass)
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
        
        //For each element of deltas,multiply by stdev
        
        var stdDev = self.boundingBoxRefinementStandardDeviation
        let stdDevPointer = UnsafeMutablePointer<Float>(&stdDev)
        elementWiseMultiply(matrixPointer: sortedDeltasPointer, vectorPointer: stdDevPointer, height:Int(numberOfElementsToProcess), width: stdDev.count)
        var resultBoxes = applyBoxDeltas(boxes: sortedAnchors, deltas: sortedDeltas)
        let resultBoxesPointer = UnsafeMutablePointer<Float>(&resultBoxes)
        clipBoxes(boxesPointer: resultBoxesPointer, elementCount: Int(numberOfElementsToProcess))
        let resultIndices = nonMaxSupression(boxes: resultBoxes,
                                             indices: Array(0 ..< sortedAnchors.count),
                                             iouThreshold: 0.7,
                                             max: 1000)
        for (i,resultIndex) in resultIndices.enumerated() {
            for j in 0 ..< 4 {
                outputs[0][i*4+j] = resultBoxes[resultIndex*4+j] as NSNumber
            }
        }
        
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

func sortProbabilities(probabilities:[Float], keepTop:UInt) -> ([Float],[Float]) {
    
    var probabilities = probabilities
    let probabilitiesPointer = UnsafeMutablePointer<Float>(&probabilities)
    let totalElements = vDSP_Length(probabilities.count)
    
    var indices = Array<vDSP_Length>(0 ..< totalElements)
    let indicesPointer = UnsafeMutablePointer<vDSP_Length>(&indices)
    vDSP_vsorti(probabilitiesPointer, indicesPointer, nil, totalElements, -1)
    
    //We clip to the limit
    let numberOfElementsToProcess = min(totalElements, keepTop)
    
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



func elementWiseMultiply(matrixPointer:UnsafeMutablePointer<Float>,
                         vectorPointer:UnsafeMutablePointer<Float>,
                         height:Int,
                         width:Int) {
    for i in 0 ..< width {
        vDSP_vsmul(matrixPointer.advanced(by: i), width, vectorPointer.advanced(by: i), matrixPointer.advanced(by: i), width, vDSP_Length(height))
    }
}
