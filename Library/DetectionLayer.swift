//
//  DetectionLayer.swift
//  Mask-RCNN-Demo
//
//  Created by Edouard Lavery-Plante on 2018-11-06.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

@objc(DetectionLayer) class DetectionLayer: NSObject, MLCustomLayer {

    let outputShapes:[[NSNumber]] = [[100,1,6,1,1]]
    let boundingBoxRefinementStandardDeviation:[Float] = [0.1, 0.1, 0.2, 0.2]

    required init(parameters: [String : Any]) throws {
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        return outputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)
        
        let log = OSLog(subsystem: "Detection", category: OSLog.Category.pointsOfInterest)
        os_signpost(.begin, log: log, name: "Detection-Eval")
        
        let rois = inputs[0]
        let roisPointer = UnsafeMutablePointer<Float>(OpaquePointer(rois.dataPointer))
        let probabilities = inputs[1] //(N,num_classes)
        let probabilitiesPointer = UnsafeMutablePointer<Float>(OpaquePointer(probabilities.dataPointer))

        let boundingBoxDeltas = inputs[2]
        let boundingBoxDeltasPointer = UnsafeMutablePointer<Float>(OpaquePointer(boundingBoxDeltas.dataPointer))

        let N = UInt(truncating:rois.shape[0])//1000
        let maxDetections = 100
        let numberOfClasses = UInt(truncating:probabilities.shape[2])//81

        var (scores,classIds) = maximumValuesWithIndices2d(values: probabilitiesPointer, rows: N, columns: numberOfClasses)

        let scoresPointer = UnsafeMutablePointer<Float>(&scores)

        //Start from all indices of rois. Omit background class and apply threshold
        
        var filteredIndices = indicesOfRoisWithHighScores(scores: UnsafeMutablePointer<Float>(&scores),
                                                          threshold: 0.3,
                                                          count: UInt(scores.count))
        filteredIndices = filteredIndices.filter { (index) -> Bool in
            let intIndex = Int(index)
            let classId = classIds[intIndex]
            return classId > 0
        }
        
        //Gather based on filtered indices
        
        let filteredRois = gather(values: roisPointer,
                                  valueSize: 4,
                                  indices: filteredIndices,
                                  indicesLength: UInt(filteredIndices.count))

        let filteredScores = gather(values: scoresPointer,
                                    valueSize: 1,
                                    indices: filteredIndices,
                                    indicesLength: UInt(filteredIndices.count))
        
        var floatClass:[Float] = Array<Float>(classIds.map({ (integerValue) -> Float in
            return Float(exactly: integerValue)!
        }))
        let floatClassPointer = UnsafeMutablePointer<Float>(&floatClass)
        let filteredClass = gather(values: floatClassPointer,
                                   valueSize: 1,
                                   indices: filteredIndices,
                                   indicesLength: UInt(filteredIndices.count))
        
        let boundingBoxDeltaIndices = deltasIndices(indices: filteredIndices, classIds: classIds)
        
        var filteredDeltas = gather(values: boundingBoxDeltasPointer,
                                    valueSize: 1,
                                    indices: boundingBoxDeltaIndices,
                                    indicesLength: UInt(boundingBoxDeltaIndices.count))
        
        var stdDev = self.boundingBoxRefinementStandardDeviation
        let stdDevPointer = UnsafeMutablePointer<Float>(&stdDev)
        elementWiseMultiply(matrixPointer: UnsafeMutablePointer<Float>(&filteredDeltas), vectorPointer: stdDevPointer, height:filteredClass.count, width: stdDev.count)

        var resultBoxes = applyBoxDeltas(boxes: filteredRois, deltas: filteredDeltas)
        resultBoxes.boxReference().clip()

        var nmsBoxIds = [Int]()
        let classIdSet = Set(filteredClass)
        
        //Apply NMS for each class that's present
        for classId in classIdSet {
            
            let indicesOfClass = filteredClass.enumerated().filter { (_,thisClassId) -> Bool in
                return thisClassId == classId
                }.map { (offset, _) -> Int in
                    return offset
            }
            
            let nmsResults = nonMaxSupression(boxes: resultBoxes,
                                              indices: indicesOfClass,
                                              iouThreshold: 0.3, max: 100)
            nmsBoxIds.append(contentsOf:nmsResults)
        }
                
        //Keep the top NOut
        let resultIndices:[Int] = {
            () -> [Int] in
            
            let maxElements = min(nmsBoxIds.count, 100)
            
            var scores = Array<Float>(repeating: 0.0, count: nmsBoxIds.count)
            
            for (i,index) in nmsBoxIds.enumerated() {
                scores[i] = filteredScores[index]
            }
            
            let zippedIdsAndScores = zip(nmsBoxIds, scores)
            
            let sortedZippedIds = zippedIdsAndScores.sorted(by: {
                (a, b) -> Bool in
                return a.1 > b.1
            })
            
            let clippedZippedIds = sortedZippedIds[0..<maxElements]
            
            return clippedZippedIds.map({ (offset, element) -> Int in
                return offset
            })
        }()
        
        let boxLength = 4
        let output = outputs[0]
        let outputElementStride = Int(truncating: output.strides[0])
        
        if(output[0] != 0) {
            print(output[0])
        }
        
        //Gather indices so that output is [N, (y1, x1, y2, x2, class_id, score)]
        
        for (i,resultIndex) in resultIndices.enumerated() {
            
            for j in 0 ..< boxLength {
                output[i*outputElementStride+j] = resultBoxes[resultIndex*boxLength+j] as NSNumber
            }
            output[i*outputElementStride+4] = filteredClass[resultIndex] as NSNumber
            output[i*outputElementStride+5] = filteredScores[resultIndex] as NSNumber
        }
        
        //Zero-pad the rest
        
        let detectionsCount = resultIndices.count
        let paddingCount = max(0,maxDetections-detectionsCount)*outputElementStride
        
        output.padTailWithZeros(startIndex: detectionsCount*outputElementStride, count: paddingCount)
        
        os_signpost(.end, log: log, name: "Detection-Eval")
    }

}

func maximumValuesWithIndices2d(values:UnsafePointer<Float>,
                                rows:UInt,
                                columns:UInt) -> ([Float],[UInt]) {
    var resultValues = Array<Float>(repeating: 0.0, count: Int(rows))
    var resultValuesPointer = UnsafeMutablePointer<Float>(&resultValues)
    var resultIndices = Array<UInt>(repeating: 0, count: Int(rows))
    var resultIndicesPointer = UnsafeMutablePointer<UInt>(&resultIndices)
    
    let columnsInt = Int(columns)
    var valuesMovingPointer = values
    
    //NOTE : We could find resultValues using vDSP_vswmax, but how would we find the index? This solution appears faster
    
    for _ in 0 ..< rows
    {
        vDSP_maxvi(valuesMovingPointer,
                   1,
                   resultValuesPointer,
                   resultIndicesPointer,
                   columns)
        resultValuesPointer = resultValuesPointer.advanced(by: 1)
        resultIndicesPointer = resultIndicesPointer.advanced(by: 1)
        valuesMovingPointer = valuesMovingPointer.advanced(by: columnsInt)
    }

    return (resultValues,resultIndices)
}

func indicesOfRoisWithHighScores(scores:UnsafeMutablePointer<Float>,
                                 threshold:Float,
                                 count:UInt) -> [Float] {
    let temporaryBuffer = UnsafeMutablePointer<Float>.allocate(capacity: Int(count))

    //Count how many scores below threshold
    var lowThreshold:Float = threshold
    var highThreshold:Float = 1.0
    var lowCount:UInt = 0
    var highCount:UInt = 0
    vDSP_vclipc(scores,
                1,
                UnsafeMutablePointer<Float>(&lowThreshold),
                UnsafeMutablePointer<Float>(&highThreshold),
                temporaryBuffer,
                1,
                count,
                UnsafeMutablePointer<UInt>(&lowCount),
                UnsafeMutablePointer<UInt>(&highCount))
    
    //Set to 0 scores below threshold
    vDSP_vthres(scores, 1, UnsafeMutablePointer<Float>(&lowThreshold), scores, 1, count)

    //Vector ramp
    var initial:Float = 0
    var increment:Float = 1
    vDSP_vramp(UnsafeMutablePointer<Float>(&initial), UnsafeMutablePointer<Float>(&increment), temporaryBuffer, 1, count)
    var indices = Array<Float>(repeating: 0, count:Int(count-lowCount))
    //Compress the ramp
    vDSP_vcmprs(temporaryBuffer,
                1,
                scores,
                1,
                UnsafeMutablePointer<Float>(&indices),
                1,
                count)
    temporaryBuffer.deallocate()
    return indices
}

func deltasIndices(indices:[Float],
                   classIds:[UInt]) -> [Float] {
    let boxLength = 4
    let numberOfClasses = 81
    let indexStride = Float(numberOfClasses*boxLength)
    var result = Array<Float>()
    
    //for each index, we will produce 4 result indices
    for (i,index) in indices.enumerated() {
        let classId = Float(classIds[i])
        let deltaIndex = index*indexStride+classId
        for j in 0 ..< boxLength {
            result.append(deltaIndex+Float(numberOfClasses*j))
        }
    }
    
    return result
}

func gather(values:UnsafePointer<Float>,
            valueSize:UInt,
            indices:[Float],
            indicesLength:UInt) -> [Float] {
    let resultCount =  Int(indicesLength*valueSize)
    var result = Array<Float>(repeating: 0.0, count:resultCount)
    let resultPointer = UnsafeMutablePointer<Float>(&result)
    
    var indicesOfValueSize = Array<Float>(repeating: 0.0, count:resultCount)
    
    for (i,index) in indices.enumerated() {
        
        for j in 0 ..< Int(valueSize) {
            indicesOfValueSize[i*Int(valueSize)+j] = index*Float(valueSize)+Float(j)
        }
        
    }
    
    let indicesOfValueSizePointer = UnsafeMutablePointer<Float>(&indicesOfValueSize)
    vDSP_vindex(values, indicesOfValueSizePointer, 1, resultPointer, 1, indicesLength)
    return result
}

