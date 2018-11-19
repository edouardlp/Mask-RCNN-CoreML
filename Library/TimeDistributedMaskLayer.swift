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
        
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)
        
        let detections = inputs[1]
        let detectionCount = Int(truncating: detections.shape[0])
        let detectionsStride = Int(truncating: detections.strides[0])
        
        let model = Mask().model
        let predictionOptions = MLPredictionOptions()
        
        //Temporary, otherwise we seem to consume all system memory
        predictionOptions.usesCPUOnly = true
        
        let batchIn = MultiArrayBatchProvider(multiArrays: inputs, featureNames: self.featureNames)
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
    }
    
}

class MultiArrayBatchProvider : MLBatchProvider
{
    let multiArrays:[MLMultiArray]
    
    var featureNames: [String]
    
    let indexMapping:[Int:Int]

    public var count: Int {
        return indexMapping.count
    }
    
    init(multiArrays:[MLMultiArray],
         featureNames:[String]) {
        
        self.multiArrays = multiArrays
        self.featureNames = featureNames
        
        var mapping = [Int:Int]()

        let stride = Int(truncating: multiArrays[0].strides[0])
        var index = 0
        var buffer = Array<Float>(repeating: 0.0, count: stride)
        let bufferPointer = UnsafeMutableRawPointer(&buffer)
        
        for i in 0 ..< Int(truncating:multiArrays[0].shape[0]) {
            bufferPointer.copyMemory(from: multiArrays[0].dataPointer.advanced(by: stride*i*4), byteCount: stride*4)
            if(buffer.allSatisfy({ (value) -> Bool in
                return value != 0
            })) {
                mapping[index] = i
                index += 1
            }
        }
        self.indexMapping = mapping
    }
    
    public func features(at index: Int) -> MLFeatureProvider {
        let mappedIndex = self.indexMapping[index]!
        return MultiArrayFeatureProvider(multiArrays: self.multiArrays, featureNames: self.featureNames, index: mappedIndex)
    }
}

class MultiArrayFeatureProvider : MLFeatureProvider
{
    let multiArrays:[MLMultiArray]
    var featureNames: Set<String> {
        return Set(orderedFeatureNames)
    }
    var orderedFeatureNames: [String]
    let index:Int
    
    init(multiArrays:[MLMultiArray],
         featureNames:[String],
         index:Int) {
        self.multiArrays = multiArrays
        self.orderedFeatureNames = featureNames
        self.index = index
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        guard let featureIndex = self.orderedFeatureNames.firstIndex(of: featureName) else
        {
            return nil
        }
        let multiArray = self.multiArrays[featureIndex]
        let outputComponentsSize = MemoryLayout<Float>.size
        guard let outputMultiArray = try? MLMultiArray(dataPointer: multiArray.dataPointer.advanced(by: Int(truncating: multiArray.strides[0])*index*outputComponentsSize), shape: Array(multiArray.shape[2...4]), dataType: multiArray.dataType, strides: Array(multiArray.strides[2...4]), deallocator: nil) else
        {
            return nil
        }
        return MLFeatureValue(multiArray: outputMultiArray)
    }
}
