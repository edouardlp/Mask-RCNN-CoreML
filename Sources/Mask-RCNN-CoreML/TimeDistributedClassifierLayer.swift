//
//  TimeDistributedClassifierLayer.swift
//  Example
//
//  Created by Edouard Lavery-Plante on 2018-11-20.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

@available(iOS 12.0, macOS 10.14, *)
@objc(TimeDistributedClassifierLayer) class TimeDistributedClassifierLayer: NSObject, MLCustomLayer {
    
    let featureNames:[String] = ["feature_map"]
    
    required init(parameters: [String : Any]) throws {
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        let featureMapShape = inputShapes[0]
        let batch = featureMapShape[1]
        let channel = featureMapShape[0]
        let outputShapes = [[channel,batch,1,1,6]]
        return outputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        
        let log = OSLog(subsystem: "TimeDistributedClassifierLayer", category: OSLog.Category.pointsOfInterest)
        os_signpost(OSSignpostType.begin, log: log, name: "TimeDistributedClassifierLayer-Eval")
        
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)

        let model = try MLModel(contentsOf:MaskRCNNConfig.defaultConfig.compiledClassifierModelURL!)
        let predictionOptions = MLPredictionOptions()
        
        let batchIn = MultiArrayBatchProvider(multiArrays: inputs, removeZeros:false, featureNames: self.featureNames)
        let batchOut = try model.predictions(from: batchIn, options: predictionOptions)
        let resultFeatureNames = ["probabilities", "bounding_boxes"]
        
        os_signpost(OSSignpostType.begin, log: log, name: "TimeDistributedClassifierLayer-ProcessOutput")

        for i in 0..<batchOut.count {
            let featureProvider = batchOut.features(at: i)
            let actualIndex = batchIn.indexMapping[i]!
            var lastIndex:Int = 0
            for (j,feature) in resultFeatureNames.enumerated() {
                
                let featureValue = featureProvider.featureValue(for: feature)
                let resultArray = featureValue!.multiArrayValue!
                assert(resultArray.dataType == MLMultiArrayDataType.double)
                let resultMemorySize = MemoryLayout<Double>.size
                let resultShape = Int(truncating: resultArray.shape[0])
                
                let outputMultiArray = outputs[0]
                let outputStride = Int(truncating: outputMultiArray.strides[2])
                
                var doubleBuffer = Array<Double>(repeating:0.0, count:resultShape)
                let doubleBufferPointer = UnsafeMutableRawPointer(&doubleBuffer)
                doubleBufferPointer.copyMemory(from: resultArray.dataPointer, byteCount: resultShape*resultMemorySize)
                
                var floatBuffer = doubleBuffer.map { (doubleValue) -> Float in
                    return Float(doubleValue)
                }
                
                //TODO: attempt to get CoreML to do this instead
        
                if(j==0){
                    let (value, index) = maximumValueWithIndex(values: floatBuffer)
                    lastIndex = Int(index)
                    //Write the argmax and score to indices 4 and 5
                    outputMultiArray[outputStride*actualIndex+4] = index as NSNumber
                    outputMultiArray[outputStride*actualIndex+5] = value as NSNumber
                } else {
                    for z in 0 ..< 4 {
                        let deltaIndex = lastIndex*4+z
                        outputMultiArray[outputStride*actualIndex+z] = floatBuffer[deltaIndex] as NSNumber
                    }
                }
            }
        }
        os_signpost(OSSignpostType.end, log: log, name: "TimeDistributedClassifierLayer-ProcessOutput")
        os_signpost(OSSignpostType.end, log: log, name: "TimeDistributedClassifierLayer-Eval")
    }
}

@available(iOS 12.0, macOS 10.14, *)
class MultiArrayBatchProvider : MLBatchProvider
{
    let multiArrays:[MLMultiArray]
    
    var featureNames: [String]
    
    let indexMapping:[Int:Int]
    
    public var count: Int {
        return indexMapping.count
    }
    
    init(multiArrays:[MLMultiArray],
         removeZeros:Bool,
         featureNames:[String]) {
        
        self.multiArrays = multiArrays
        self.featureNames = featureNames
        var mapping = [Int:Int]()
        let stride = Int(truncating: multiArrays[0].strides[0])
        var index = 0
        if(removeZeros){
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
        } else {
            for i in 0 ..< Int(truncating:multiArrays[0].shape[0]) {
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

@available(iOS 12.0, macOS 10.14, *)
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


func maximumValueWithIndex(values:[Float]) -> (Float,UInt) {
    
    var values = values
    var resultValue:Float = 0.0
    let resultValuePointer = UnsafeMutablePointer<Float>(&resultValue)
    var resultIndex:UInt = 0
    let resultIndexPointer = UnsafeMutablePointer<UInt>(&resultIndex)
    
    vDSP_maxvi(UnsafeMutablePointer<Float>(&values),
               1,
               resultValuePointer,
               resultIndexPointer,
               vDSP_Length(values.count))
    
    return (resultValue,resultIndex)
}
