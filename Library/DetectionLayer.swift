//
//  DetectionLayer.swift
//  Mask-RCNN-CoreML
//
//  Created by Edouard Lavery-Plante on 2018-11-06.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

/**

 DetectionLayer is a Custom ML Layer that outputs object detections.
 
 DetectionLayer outputs detections based on the confidence of an object
 (any non-background class) being detected in each region.
 Regions that overlap more than a given threshold are removed
 through a process called Non-Max Supression (NMS).
 
 The regions of interest are adjusted using deltas provided as input to refine
 how they enclose the detected objects.
 
 The layer takes three inputs :
 
 - Regions of interest. Shape : (#regions,4)
 - TODO
 
 The regions of interest are layed out as follows : (y1,x1,y2,x2).
 
 The class probability input is layed out such that the index 0 corresponds to the
 background class.
 
 The bounding box deltas are layed out as follows : (dy,dx,log(dh),log(dw)).
 
 The layer takes four parameters :
 
 - boundingBoxRefinementStandardDeviation : Bounding box deltas refinement standard deviation
 - detectionLimit : Maximum # of detections to output
 - lowConfidenceScoreThreshold : Threshold below which to discard regions
 - nmsIOUThreshold : Threshold below which to supress regions that overlap
 
 The layer has one ouput :
 
 - Detections (y1,x1,y2,x2,classId,score). Shape : (#regionsOut,6)
 
The classIds are the argmax of each rows in the class probability input.
 
 */
@objc(DetectionLayer) class DetectionLayer: NSObject, MLCustomLayer {
    
    //Bounding box deltas refinement standard deviation
    var boundingBoxRefinementStandardDeviation:[Float] = [0.1, 0.1, 0.2, 0.2]
    //Maximum # of detections to output
    var maxDetections = 100
    //Threshold below which to discard regions
    var lowConfidenceScoreThreshold:Float = 0.7
    //Threshold below which to supress regions that overlap
    var nmsIOUThreshold:Float = 0.3
    
    required init(parameters: [String : Any]) throws {
        
        super.init()
        
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
        
        if let maxDetections = parameters["maxDetections"] as? Int {
            self.maxDetections = maxDetections
        }
        if let lowConfidenceScoreThreshold = parameters["scoreThreshold"] as? Double {
            self.lowConfidenceScoreThreshold = Float(lowConfidenceScoreThreshold)
        }
        if let nmsIOUThreshold = parameters["nmsIOUThreshold"] as? Double {
            self.nmsIOUThreshold = Float(nmsIOUThreshold)
        }
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {

        let roisShape = inputShapes[0]
        
        let seq = maxDetections as NSNumber
        let batch = roisShape[1]
        let channels:NSNumber = 6//(y1,x1,y2,x2,classId,score)
        let height:NSNumber = 1
        let width:NSNumber = 1
        let outputShapes = [[seq,batch,channels,height,width]]
        return outputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {

        let log = OSLog(subsystem: "DetectionLayer", category: OSLog.Category.pointsOfInterest)
        os_signpost(OSSignpostType.begin, log: log, name: "Detection-Eval")
        
        //Regions of interest. Shape : (#regions,4)
        //(y1,x1,y2,x2)
        let rois = inputs[0]
        assert(inputs[0].dataType == MLMultiArrayDataType.float32)

        //TODO: doc
        //dy,dx,log(dh),log(dw),classId,score)
        let classifications = inputs[1]
        assert(inputs[1].dataType == MLMultiArrayDataType.float32)
        
        let inputRegionsCount = Int(truncating:rois.shape[0])
        let maxDetections = self.maxDetections
        
        var boundingBoxDeltas = classifications.floatDataPointer().stridedSlice(begin: 0, count: inputRegionsCount, stride: 6, length: 4)
        
        var classIds = classifications.floatDataPointer().stridedSlice(begin: 4, count: inputRegionsCount, stride: 6)
        var scores = classifications.floatDataPointer().stridedSlice(begin: 5, count: inputRegionsCount, stride: 6)

        //Start from all indices of rois and apply threshold
        var filteredIndices = indicesOfRoisWithHighScores(scores: UnsafeMutablePointer<Float>(&scores),
                                                          threshold: self.lowConfidenceScoreThreshold,
                                                          count: UInt(scores.count))
        
        //Omit background class
        filteredIndices = filteredIndices.filter { (index) -> Bool in
            let intIndex = Int(index)
            let classId = classIds[intIndex]
            return classId > 0
        }
        
        //Gather rois based on filtered indices
        let boxElementLength = 4
        let boxIndices = broadcastedIndices(indices: filteredIndices, toElementLength: boxElementLength)
        var filteredRois = rois.floatDataPointer().indexed(indices: boxIndices)
        
        //Gather scores based on filtered indices
        let filteredScores = UnsafeMutablePointer<Float>(&scores).indexed(indices:filteredIndices)
        
        //Gather classes based on filtered indices
        let filteredClass = UnsafeMutablePointer<Float>(&classIds).indexed(indices:filteredIndices)
     
        //Gather deltas based on filtered indices
        var filteredDeltas =  UnsafeMutablePointer<Float>(&boundingBoxDeltas).indexed(indices: boxIndices)
        
        //Multiply bounding box deltas by std dev
        var stdDev = self.boundingBoxRefinementStandardDeviation
        let stdDevPointer = UnsafeMutablePointer<Float>(&stdDev)
        elementWiseMultiply(matrixPointer: UnsafeMutablePointer<Float>(&filteredDeltas), vectorPointer: stdDevPointer, height:filteredClass.count, width: stdDev.count)

        //Apply deltas and clip rois
        let roisReference = filteredRois.boxReference()
        roisReference.applyBoxDeltas(filteredDeltas)
        roisReference.clip()

        var nmsBoxIds = [Int]()
        let classIdSet = Set(filteredClass)
        
        //Apply NMS for each class that's present
        for classId in classIdSet {
            
            let indicesOfClass = filteredClass.enumerated().filter { (_,thisClassId) -> Bool in
                return thisClassId == classId
                }.map { (offset, _) -> Int in
                    return offset
            }
            
            let nmsResults = nonMaxSupression(boxes: filteredRois,
                                              indices: indicesOfClass,
                                              iouThreshold: self.nmsIOUThreshold,
                                              max: self.maxDetections)
            nmsBoxIds.append(contentsOf:nmsResults)
        }
                
        //Keep the top NOut
        let resultIndices:[Int] = {
            () -> [Int] in
            
            let maxElements = min(nmsBoxIds.count, self.maxDetections)
            
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
        
        //Layout output is [NOut, (y1, x1, y2, x2, class_id, score)]
        
        for (i,resultIndex) in resultIndices.enumerated() {
            
            for j in 0 ..< boxLength {
                output[i*outputElementStride+j] = filteredRois[resultIndex*boxLength+j] as NSNumber
            }
            output[i*outputElementStride+4] = filteredClass[resultIndex] as NSNumber
            output[i*outputElementStride+5] = filteredScores[resultIndex] as NSNumber
        }
        
        //Zero-pad the rest as CoreML does not erase the memory between evaluations
        
        let detectionsCount = resultIndices.count
        let paddingCount = max(0,maxDetections-detectionsCount)*outputElementStride
        
        output.padTailWithZeros(startIndex: detectionsCount*outputElementStride, count: paddingCount)
        
        os_signpost(OSSignpostType.end, log: log, name: "Detection-Eval")
    }

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
