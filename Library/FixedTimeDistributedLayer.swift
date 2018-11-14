//
//  FixedTimeDistributedLayer.swift
//  Mask-RCNN-Demo
//
//  Created by Edouard Lavery-Plante on 2018-10-31.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

@objc(FixedTimeDistributedLayer) class FixedTimeDistributedLayer: NSObject, MLCustomLayer {

    let featureNames:[String] = ["pooled_region"]
    let outputShapes:[[NSNumber]] = [[1,1,100,28,28]]
    
    required init(parameters: [String : Any]) throws {
        super.init()
        //TODO: dynamically load model
        //TODO: featureNames and outputshapes
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        return outputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)
        return
        let model = Mask().model
        let predictionOptions = MLPredictionOptions()
        predictionOptions.usesCPUOnly = true
        let batchIn = MultiArrayBatchProvider(multiArrays: inputs, featureNames: self.featureNames)
        let batchOut = try model.predictions(from: batchIn, options: predictionOptions)
        let resultFeatureNames = ["mask"]
        for i in 0..<batchOut.count {
            let featureProvider = batchOut.features(at: i)
            let actualIndex = batchIn.indexMapping[i]!
            for (j,feature) in resultFeatureNames.enumerated()
            {
                let featureValue = featureProvider.featureValue(for: feature)
                let outputMultiArray = outputs[j]
                let stride = Int(truncating: outputMultiArray.strides[2])
                let resultArray = featureValue!.multiArrayValue!
                outputMultiArray.dataPointer.advanced(by: stride*actualIndex*4).copyMemory(from: resultArray.dataPointer, byteCount: stride*4)
                
            }
        }
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
        guard let outputMultiArray = try? MLMultiArray(dataPointer: multiArray.dataPointer.advanced(by: Int(truncating: multiArray.strides[0])*index*4), shape: Array(multiArray.shape[2...4]), dataType: multiArray.dataType, strides: Array(multiArray.strides[2...4]), deallocator: nil) else
        {
            return nil
        }
        return MLFeatureValue(multiArray: outputMultiArray)
    }
}
