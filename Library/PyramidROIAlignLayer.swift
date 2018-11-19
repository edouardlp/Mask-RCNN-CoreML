//
//  PyramidROIAlignLayer.swift
//  Mask-RCNN-CoreML
//
//  Created by Edouard Lavery-Plante on 2018-10-31.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML
import Accelerate
import MetalPerformanceShaders

/**
 
 PyramidROIAlignLayer is a Custom ML Layer that extracts feature maps based on the regions of interest.
 
 PyramidROIAlignLayer outputs aligned feature maps by cropping and
 resizing portions of the input feature maps based on the regions of interest.
 
 The region's size determine which input feature map is used.
 
  The layer takes five inputs :
 - Regions of interest. Shape (#regions, 4)
 - 4 feature maps of different sizes. Shapes : (#channels,256,256), (#channels,128,128),
 (#channels,64,64),(#channels,32,32)
 
 The layer takes two parameters :
 - poolSize : The size of the output feature map
 - imageSize : The input image sizes
 
 The imageSize is used to determine which feature map to use

 The layer has one output
 - Feature maps. Shape (#regions, 1, #channels, poolSize, poolSize)

 */
@objc(PyramidROIAlignLayer) class PyramidROIAlignLayer: NSObject, MLCustomLayer {
    
    let maxBatchSize = 32//TODO: compute based on metal buffer sizes
    var poolSize:Int = 7
    var imageSize = CGSize(width: 1024, height: 1024)
    
    required init(parameters: [String : Any]) throws {
        super.init()
        
        if let poolSize = parameters["poolSize"] as? Int {
            self.poolSize = poolSize
        }
        
        if let imageWidth = parameters["imageWidth"] as? CGFloat,
           let imageHeight = parameters["imageHeight"] as? CGFloat {
            self.imageSize = CGSize(width: imageWidth, height: imageHeight)
        }
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        
        let roisShape = inputShapes[0]
        let featureMapShape = inputShapes[1]
        
        let seq = roisShape[0]
        let batch = roisShape[1]
        let channel = featureMapShape[2]
        let height = self.poolSize as NSNumber
        let width = self.poolSize as NSNumber
        let outputShapes = [[seq,batch,channel,height,width]]
        return outputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)
        
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue:MTLCommandQueue = device.makeCommandQueue(maxCommandBufferCount: 1)!

        let rois = inputs[0]
        let featureMaps = inputs[1..<inputs.count]
        
        let featureMapShape = featureMaps.first!.shape
        
        let channels = Int(truncating: featureMapShape[2])
        let floatSize = MemoryLayout<Float>.size
        
        let outputWidth = self.poolSize
        let outputHeight = self.poolSize
        
        let batches = roisToMPSRegionBatches(rois: rois,
                                             maxBatchSize: self.maxBatchSize,
                                             featureMapSelectionFactor: 224,
                                             imageSize: self.imageSize)
        
        if(batches.isEmpty) {
            return
        }
        
        let featureMapImages = featureMaps.map {
            (featureMap) -> MPSImage in
            let inputWidth = Int(truncating: featureMap.shape[4])
            let inputHeight = Int(truncating: featureMap.shape[3])
            let image = MPSImage(device: device, imageDescriptor: MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float32, width: inputWidth, height: inputHeight, featureChannels: channels, numberOfImages:1, usage:MTLTextureUsage.shaderRead))
            image.writeBytes(featureMap.dataPointer, dataLayout: MPSDataLayout.featureChannelsxHeightxWidth, imageIndex: 0)
            return image
        }
        
        let outputPointer = UnsafeMutablePointer<Float>(OpaquePointer(outputs[0].dataPointer))
        
        let resultStride = Int(truncating: outputs[0].strides[0])
        let metalRegion = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                    size: MTLSize(width: outputWidth, height: outputHeight, depth: 1))
        
        //We write to the same images to improve performance
        var outputImages = Array<MPSImage>()
        for _ in 0 ..< self.maxBatchSize {
            outputImages.append(MPSImage(device: commandQueue.device, imageDescriptor: MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float32, width: outputWidth, height: outputHeight, featureChannels: channels)))
        }

        for batch in batches {
            
            let batchOutputs = performBatch(commandQueue: commandQueue,
                                            featureMaps: featureMapImages,
                                            channelsSize:channels,
                                            outputWidth: outputWidth,
                                            outputHeight: outputHeight,
                                            batch: batch,
                                            outputImages:outputImages)
            
            for output in batchOutputs {
                
                let offset = output.offset * resultStride
                let pointer = outputPointer.advanced(by:offset)
                output.outputImage.readBytes(pointer,
                                             dataLayout: MPSDataLayout.featureChannelsxHeightxWidth,
                                             bytesPerRow: outputWidth*floatSize,
                                             region: metalRegion, featureChannelInfo: MPSImageReadWriteParams(featureChannelOffset: 0, numberOfFeatureChannelsToReadWrite: channels), imageIndex: 0)
            }
            //TODO: add zeros where a batch has invalid regions

        }
    }
    
}

func performBatch(commandQueue:MTLCommandQueue,
                  featureMaps:[MPSImage],
                  channelsSize:Int,
                  outputWidth:Int,
                  outputHeight:Int,
                  batch:MPSRegionBatchInput,
                  outputImages:[MPSImage]) -> [MPSRegionBatchOutput] {
    
    let buffer = commandQueue.makeCommandBufferWithUnretainedReferences()!//
    let channels = channelsSize
    let inputImage = featureMaps[batch.featureMapIndex]
    
    var batchOutputs = [MPSRegionBatchOutput]()
    
    for (r,region) in batch.regions.enumerated() {
        
        let outputImage = outputImages[r]
        let output = MPSRegionBatchOutput(offset: batch.offset+r, outputImage: outputImage)
        batchOutputs.append(output)
        var regions = [region]
        let regionsPointer = UnsafeMutablePointer<MPSRegion>(&regions)
        let kernel = MPSNNCropAndResizeBilinear(device: commandQueue.device,
                                                resizeWidth: outputWidth,
                                                resizeHeight: outputHeight,
                                                numberOfRegions: 1,
                                                regions: regionsPointer)
        
        for slice in 0 ..< channels/4 {
            kernel.sourceFeatureChannelOffset = slice*4
            kernel.destinationFeatureChannelOffset = slice*4
            kernel.encode(commandBuffer: buffer,
                          sourceImage: inputImage,
                          destinationImage: outputImage)
        }
    }
    
    buffer.commit()
    buffer.waitUntilCompleted()
    return batchOutputs
}

struct MPSRegionBatchInput {
    let offset:Int
    let featureMapIndex:Int
    var regions:[MPSRegion] = []
}

struct MPSRegionBatchOutput {
    let offset:Int
    let outputImage:MPSImage
}

func roisToMPSRegionBatches(rois:MLMultiArray,
                            maxBatchSize:Int,
                            featureMapSelectionFactor:CGFloat,
                            imageSize:CGSize) -> [MPSRegionBatchInput] {
    
    let totalCount = Int(truncating: rois.shape[0])
    var currentBatch:MPSRegionBatchInput?
    var result = [MPSRegionBatchInput]()
    let stride = Int(truncating: rois.strides[0])
    let ratio = Double(featureMapSelectionFactor/sqrt(imageSize.width*imageSize.height))
    
    var mapLevels = [Int]()
    
    for i in 0 ..< totalCount {
        
        let roiIndex = stride*i
        
        let y1 = Double(truncating: rois[roiIndex])
        let x1 = Double(truncating: rois[roiIndex+1])
        let y2 = Double(truncating: rois[roiIndex+2])
        let x2 = Double(truncating: rois[roiIndex+3])

        let width = x2-x1
        let height = y2-y1
        
        let featureMapLevelFloat = log2(sqrt(width*height)/ratio)+4.0
        let regionIsValid = !featureMapLevelFloat.isNaN && !featureMapLevelFloat.isInfinite
        
        let featureMapLevel = (!regionIsValid) ? 2 : min(5,max(2,Int(round(featureMapLevelFloat))))
        mapLevels.append(featureMapLevel)
        
        let featureMapIndex = featureMapLevel-2

        let region = MPSRegion(origin: MPSOrigin(x: x1, y: y1, z: 0), size: MPSSize(width: width,
                                                                                    height: height,
                                                                                    depth: 1.0))
        
        if let existingBatch = currentBatch {
            if(existingBatch.regions.count == maxBatchSize ||
                featureMapIndex != existingBatch.featureMapIndex || !regionIsValid) {
                //Close the batch
                result.append(existingBatch)
                if(regionIsValid) {
                    //Only open a new batch if the new region is valid
                    currentBatch = MPSRegionBatchInput(offset: i, featureMapIndex: featureMapIndex, regions: [region])
                } else {
                    currentBatch = nil
                }
            }
            else {
                currentBatch!.regions.append(region)
            }
        }
        else {
            currentBatch = MPSRegionBatchInput(offset: i, featureMapIndex: featureMapIndex, regions: [region])
        }
    }
    
    if let currentBatch = currentBatch {
        result.append(currentBatch)
    }
    
    return result
}
