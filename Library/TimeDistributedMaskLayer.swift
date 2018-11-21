//
//  TimeDistributedMask.swift
//  Mask-RCNN-Demo
//
//  Created by Edouard Lavery-Plante on 2018-10-31.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

@objc(TimeDistributedMaskLayer) class TimeDistributedMaskLayer: NSObject, MLCustomLayer {

    let featureNames:[String] = ["feature_map"]
    
    required init(parameters: [String : Any]) throws {
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        let featureMapShape = inputShapes[0]
        let poolHeight = featureMapShape[3]
        let poolWidth = featureMapShape[4]
        let seq = 1 as NSNumber
        let batch = featureMapShape[1]
        let channel = featureMapShape[0]
        let height = Int(truncating: poolHeight)*2 as NSNumber
        let width = Int(truncating: poolWidth)*2 as NSNumber
        let outputShapes = [[seq,batch,channel,height,width]]
        return outputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        
        let log = OSLog(subsystem: "TimeDistributedMaskLayer", category: OSLog.Category.pointsOfInterest)
        os_signpost(OSSignpostType.begin, log: log, name: "TimeDistributedMask-Eval")
        
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)
        let detections = inputs[1]
        let detectionCount = Int(truncating: detections.shape[0])
        let detectionsStride = Int(truncating: detections.strides[0])
        
        let model = Mask().model
        let predictionOptions = MLPredictionOptions()
        
        let batchIn = MultiArrayBatchProvider(multiArrays: inputs, removeZeros:true, featureNames: self.featureNames)
        let batchOut = try model.predictions(from: batchIn, options: predictionOptions)
        let resultFeatureNames = ["masks"]
        
        let output = outputs[0]
        let outputStride = Int(truncating: output.strides[2])
        
        for i in 0..<batchOut.count {
            let featureProvider = batchOut.features(at: i)
            let actualIndex = batchIn.indexMapping[i]!
            for (j,feature) in resultFeatureNames.enumerated()
            {
                let featureValue = featureProvider.featureValue(for: feature)
                let outputMultiArray = outputs[j]
                let stride = Int(truncating: outputMultiArray.strides[2])
                let resultArray = featureValue!.multiArrayValue!
                assert(resultArray.dataType == MLMultiArrayDataType.double)
                let resultMemorySize = MemoryLayout<Double>.size
                let resultStride = Int(truncating: resultArray.strides[0])

                let classId = Int(truncating: detections[detectionsStride*i+4])
                
                var doubleBuffer = Array<Double>(repeating:0.0, count:stride)
                let doubleBufferPointer = UnsafeMutableRawPointer(&doubleBuffer)
                doubleBufferPointer.copyMemory(from: resultArray.dataPointer.advanced(by: resultStride*classId*resultMemorySize), byteCount: stride*resultMemorySize)
                
                var floatBuffer = doubleBuffer.map { (doubleValue) -> Float in
                    return Float(doubleValue)
                }
                let floatBufferPointer = UnsafeMutableRawPointer(&floatBuffer)

                let outputMemorySize = MemoryLayout<Float>.size
                outputMultiArray.dataPointer.advanced(by: stride*actualIndex*outputMemorySize).copyMemory(from: floatBufferPointer, byteCount: stride*outputMemorySize)
            }
        }
        
        let resultCount = batchOut.count
        let paddingCount = max(0,detectionCount-resultCount)*outputStride
        output.padTailWithZeros(startIndex: resultCount*outputStride, count: paddingCount)
        os_signpost(OSSignpostType.end, log: log, name: "TimeDistributedMask-Eval")
    }
}
