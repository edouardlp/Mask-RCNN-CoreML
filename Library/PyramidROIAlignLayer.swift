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
import os.signpost

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
        
        let log = OSLog(subsystem: "PyramidROIAlign", category: OSLog.Category.pointsOfInterest)
        os_signpost(OSSignpostType.begin, log: log, name: "PyramidROIAlign-Eval")
        
        let device = MTLCreateSystemDefaultDevice()!
        let commandQueue:MTLCommandQueue = device.makeCommandQueue(maxCommandBufferCount: 1)!

        let rois = inputs[0]
        let featureMaps = inputs[1..<inputs.count]
        
        let featureMapShape = featureMaps.first!.shape
        
        let channels = Int(truncating: featureMapShape[2])
        let floatSize = MemoryLayout<Float>.size
        
        let outputWidth = self.poolSize
        let outputHeight = self.poolSize
        
        let items = roisToInputItems(rois: rois, featureMapSelectionFactor: 224, imageSize: self.imageSize)
        let groups = groupInputItemsByContent(items: items)
        
        let batches = batchInputGroups(groups: groups, maxComputeBatchSize: self.maxBatchSize)
        
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
        
        let outputPointer = outputs[0].dataPointer
        
        let resultStride = Int(truncating: outputs[0].strides[0])
        let metalRegion = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                                    size: MTLSize(width: outputWidth, height: outputHeight, depth: 1))
        
        //We write to the same images to improve performance
        var outputImages = Array<MPSImage>()
        for _ in 0 ..< self.maxBatchSize {
            outputImages.append(MPSImage(device: commandQueue.device, imageDescriptor: MPSImageDescriptor(channelFormat: MPSImageFeatureChannelFormat.float32, width: outputWidth, height: outputHeight, featureChannels: channels)))
        }
        
        var paddingBuffer = Array<Float>(repeating: 0.0, count: self.maxBatchSize)

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
                let pointer = outputPointer.advanced(by:offset*MemoryLayout<Float>.size)
                switch(output.content){
                case .image(image: let image):
                    image.readBytes(pointer,
                                    dataLayout: MPSDataLayout.featureChannelsxHeightxWidth,
                                    bytesPerRow: outputWidth*floatSize,
                                    region: metalRegion, featureChannelInfo: MPSImageReadWriteParams(featureChannelOffset: 0, numberOfFeatureChannelsToReadWrite: channels), imageIndex: 0)
                case .padding(count: let paddingCount):
                    if(paddingCount > paddingBuffer.count){
                        paddingBuffer = Array<Float>(repeating: 0.0, count: paddingCount)
                    }
                    
                    let bufferPointer = UnsafeMutableRawPointer(&paddingBuffer)
                    pointer.copyMemory(from: bufferPointer, byteCount: MemoryLayout<Float>.size*paddingCount)
                }
            }

        }
        os_signpost(OSSignpostType.end, log: log, name: "PyramidROIAlign-Eval")
    }
    
}

func performBatch(commandQueue:MTLCommandQueue,
                  featureMaps:[MPSImage],
                  channelsSize:Int,
                  outputWidth:Int,
                  outputHeight:Int,
                  batch:ROIAlignInputBatch,
                  outputImages:[MPSImage]) -> [ROIAlignOutputItem] {
    
    let buffer = commandQueue.makeCommandBufferWithUnretainedReferences()!
    let channels = channelsSize
    
    var outputItems = [ROIAlignOutputItem]()
    var computeOffset:Int = 0

    for group in batch.groups {
        switch(group.content){
        case .regions(featureMapIndex: let featureMapIndex, regions: let regions):
            let inputImage = featureMaps[featureMapIndex]
            for (r,region) in regions.enumerated() {
                
                let outputImage = outputImages[computeOffset]
                let output = ROIAlignOutputItem(offset: group.offset+r, content: .image(image: outputImage))
                outputItems.append(output)
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
                computeOffset += 1
            }
        case .padding(count: let paddingCount):
            let output = ROIAlignOutputItem(offset: group.offset, content: .padding(count: paddingCount))
            outputItems.append(output)
        }
    }
    
    
    buffer.commit()
    buffer.waitUntilCompleted()
    return outputItems
}

struct ROIAlignInputBatch {
    
    let groups:[ROIAlignInputGroup]
    
    var size:Int {
        return self.groups.reduce(0, { (result, group) -> Int in
            return result + group.size
        })
    }
    
    var computeSize:Int {
        return self.groups.reduce(0, { (result, group) -> Int in
            return result + group.computeSize
        })
    }
    
}

struct ROIAlignInputGroup {
    
    enum Content {
        case regions(featureMapIndex:Int, regions:[MPSRegion])
        case padding(count:Int)
    }
    
    let offset:Int
    let content:Content
    
    var size:Int {
        switch self.content {
        case .padding(count: let count):
            return count
        default:
            return self.computeSize
        }
    }
    
    var computeSize:Int {
        switch self.content {
        case .regions(featureMapIndex: _, regions: let regions):
            return regions.count
        default:
            return 0
        }
    }
    
}

struct ROIAlignInputItem {
    
    enum Content {
        case region(featureMapIndex:Int, region:MPSRegion)
        case padding
    }
    
    let offset:Int
    let content:Content
}

struct ROIAlignOutputItem {
    
    enum Content {
        case image(image:MPSImage)
        case padding(count:Int)
    }
    
    let offset:Int
    let content:Content
}

func roisToInputItems(rois:MLMultiArray,
                      featureMapSelectionFactor:CGFloat,
                      imageSize:CGSize) -> [ROIAlignInputItem] {
    
    let totalCount = Int(truncating: rois.shape[0])
    let stride = Int(truncating: rois.strides[0])
    let ratio = Double(featureMapSelectionFactor/sqrt(imageSize.width*imageSize.height))
    
    var results = [ROIAlignInputItem]()
    
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
        let featureMapIndex = featureMapLevel-2

        let region = MPSRegion(origin: MPSOrigin(x: x1, y: y1, z: 0), size: MPSSize(width: width,
                                                                                    height: height,
                                                                                    depth: 1.0))
        
        let content:ROIAlignInputItem.Content = {
            () -> ROIAlignInputItem.Content in
            if(regionIsValid){
                return .region(featureMapIndex: featureMapIndex, region: region)
            }
            return .padding
        }()
        
        let item = ROIAlignInputItem(offset: i, content: content)
        results.append(item)
    }

    return results
}

func groupInputItemsByContent(items:[ROIAlignInputItem]) -> [ROIAlignInputGroup] {
    
    var results = [ROIAlignInputGroup]()
    
    var offset:Int = 0
    var currentPaddingCount:Int?
    var currentFeatureMapIndex:Int?
    var currentRegions:[MPSRegion]?
    
    let closeRegionsGroupIfNecessary = {
        
        () -> Void in
        
        if let mapIndex = currentFeatureMapIndex, let regions = currentRegions {
            let group = ROIAlignInputGroup(offset: offset, content: ROIAlignInputGroup.Content.regions(featureMapIndex: mapIndex, regions: regions))
            results.append(group)
            currentFeatureMapIndex = nil
            currentRegions = nil
            offset += group.size
        }
    }
    
    let closePaddingGroupIfNecessary = {
        
        () -> Void in
        if let paddingCount = currentPaddingCount {
            let group = ROIAlignInputGroup(offset: offset, content: ROIAlignInputGroup.Content.padding(count: paddingCount))
            results.append(group)
            currentPaddingCount = nil
            offset += group.size
        }
    }
    
    for item in items {
        
        switch(item.content)
        {
        case .padding:
            
            closeRegionsGroupIfNecessary()
            
            if let paddingCount = currentPaddingCount {
                currentPaddingCount = paddingCount + 1
            } else {
                currentPaddingCount = 1
            }
            
        case .region(featureMapIndex: let mapIndex, region: let region):
            
            closePaddingGroupIfNecessary()
            
            if let currentFeatureMapIndex = currentFeatureMapIndex, currentFeatureMapIndex == mapIndex  {
                currentRegions?.append(region)
            } else {
                closeRegionsGroupIfNecessary()
                currentFeatureMapIndex = mapIndex
                currentRegions = [region]
            }
            
        }
        
    }
    
    return results
}

func batchInputGroups(groups:[ROIAlignInputGroup], maxComputeBatchSize:Int) -> [ROIAlignInputBatch] {
    
    var batches = [ROIAlignInputBatch]()
    
    var groupsInBatch = [ROIAlignInputGroup]()
    var currentComputeSize:Int = 0
    
    for group in groups {
        
        let nextComputeSize = currentComputeSize + group.computeSize
        
        if(nextComputeSize <= maxComputeBatchSize) {
            groupsInBatch.append(group)
            currentComputeSize = nextComputeSize
        } else {
            batches.append(ROIAlignInputBatch(groups: groupsInBatch))
            groupsInBatch = [group]
            currentComputeSize = group.computeSize
        }
        
    }
    
    if(!groupsInBatch.isEmpty) {
        batches.append(ROIAlignInputBatch(groups: groupsInBatch))
    }
    
    return batches
    
}
