//
//  BoxUtils.swift
//  Mask-RCNN-Demo
//
//  Created by Edouard Lavery-Plante on 2018-11-06.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import Accelerate

func applyBoxDeltas(boxes:[Float],
                    deltas:[Float]) -> [Float] {
    
    precondition(boxes.count == deltas.count)
    
    var results:[Float] = Array(repeating: 0.0, count: boxes.count)
    
    for i in 0 ..< boxes.count/4 {
        
        let y1 = boxes[i*4]
        let x1 = boxes[i*4+1]
        let y2 = boxes[i*4+2]
        let x2 = boxes[i*4+3]
        
        let deltaY1 = deltas[i*4]
        let deltaX1 = deltas[i*4+1]
        let deltaY2 = deltas[i*4+2]
        let deltaX2 = deltas[i*4+3]
        
        var height = y2 - y1
        var width = x2 - x1
        var centerY = y1 + 0.5 * height
        var centerX = x1 + 0.5 * width
        
        centerY += deltaY1 * height
        centerX += deltaX1 * width
        
        height *= exp(deltaY2)
        width *= exp(deltaX2)
        
        let resultY1 = centerY - 0.5 * height
        let resultX1 = centerX - 0.5 * width
        let resultY2 = resultY1 + height
        let resultX2 = resultX1 + width
        
        results[i*4] = resultY1
        results[i*4+1] = resultX1
        results[i*4+2] = resultY2
        results[i*4+3] = resultX2
    }
    
    return results
}

func applyBoxDeltasAccelerate(boxesPointer:UnsafeMutablePointer<Float>,
                    deltasPointer:UnsafeMutablePointer<Float>,
                    elementCount:Int) {
    let boxSize = 4
    var halfScalar:Float = 0.5
    let halfScalarPointer = UnsafeMutablePointer<Float>(&halfScalar)
    
    var height = Array<Float>(repeating: 0, count: elementCount)
    let heightPointer = UnsafeMutablePointer<Float>(&height)
    var width = Array<Float>(repeating: 0, count: elementCount)
    let widthPointer = UnsafeMutablePointer<Float>(&width)
    var centerY = Array<Float>(repeating: 0, count: elementCount)
    let centerYPointer = UnsafeMutablePointer<Float>(&centerY)
    var centerX = Array<Float>(repeating: 0, count: elementCount)
    let centerXPointer = UnsafeMutablePointer<Float>(&centerX)
    
    //height = boxes[2] - boxes[0]
    vDSP_vsub(boxesPointer, boxSize, boxesPointer.advanced(by: 2), boxSize, heightPointer, 1, vDSP_Length(elementCount))
    //width = boxes[3] - boxes[1]
    vDSP_vsub(boxesPointer.advanced(by: 1), boxSize, boxesPointer.advanced(by: 3), boxSize, widthPointer, 1, vDSP_Length(elementCount))
    //center_y = boxes[0] + 0.5 * height
    vDSP_vsmul(heightPointer, 1, halfScalarPointer, centerYPointer, 1, vDSP_Length(elementCount))
    vDSP_vadd(boxesPointer, boxSize, centerYPointer, 1, centerYPointer, 1, vDSP_Length(elementCount))
    //center_x = boxes[1] + 0.5 * width
    vDSP_vsmul(widthPointer, 1, halfScalarPointer, centerXPointer, 1, vDSP_Length(elementCount))
    vDSP_vadd(boxesPointer.advanced(by: 1), boxSize, centerXPointer, 1, centerXPointer, 1, vDSP_Length(elementCount))
    
    //Apply the deltas
    
    //center_y += deltas[0] * height
    vDSP_vma(deltasPointer, boxSize, heightPointer, 1, centerYPointer, 1, centerYPointer, 1, vDSP_Length(elementCount))
    //center_x += deltas[1] * width
    vDSP_vma(deltasPointer.advanced(by: 1), boxSize, widthPointer, 1, centerXPointer, 1, centerXPointer, 1, vDSP_Length(elementCount))
    
    var deltaHeight = Array<Float>(repeating: 0, count: elementCount)
    let deltaHeightPointer = UnsafeMutablePointer<Float>(&deltaHeight)
    cblas_scopy(Int32(elementCount), deltasPointer.advanced(by: 2), Int32(boxSize), deltaHeightPointer, 1)
    
    var deltaWidth = Array<Float>(repeating: 0, count: elementCount)
    let deltaWidthPointer = UnsafeMutablePointer<Float>(&deltaWidth)
    cblas_scopy(Int32(elementCount), deltasPointer.advanced(by: 3), Int32(boxSize), deltaWidthPointer, 1)
    
    var exponentiationCount = Int32(elementCount)
    let exponentiationCountPointer = UnsafeMutablePointer<Int32>(&exponentiationCount)
    
    var temporaryHeight = Array<Float>(repeating: 0, count: elementCount)
    let temporaryHeightPointer = UnsafeMutablePointer<Float>(&temporaryHeight)
    var temporaryWidth = Array<Float>(repeating: 0, count: elementCount)
    let temporaryWidthPointer = UnsafeMutablePointer<Float>(&temporaryWidth)
    
    //height *= exp(deltas[2])
    vvexpf(temporaryHeightPointer, deltaHeight, exponentiationCountPointer)
    vDSP_vsmul(heightPointer, 1, temporaryHeightPointer, heightPointer, 1, vDSP_Length(elementCount))
    
    //width *= exp(deltas[3])
    vvexpf(temporaryWidthPointer, deltaWidth, exponentiationCountPointer)
    vDSP_vsmul(widthPointer, 1, temporaryWidthPointer, widthPointer, 1, vDSP_Length(elementCount))
    
    //Convert back to normalized coordinates y1, x1, y2, x2 and store results in boxes pointer
    
    //boxes[0](y1) = center_y - 0.5 * height
    vDSP_vsmul(heightPointer, 1, halfScalarPointer, temporaryHeightPointer, 1, vDSP_Length(elementCount))
    vDSP_vsub(temporaryHeightPointer, 1, centerYPointer, 1, boxesPointer, boxSize, vDSP_Length(elementCount))
    //boxes[1](x1) = center_x - 0.5 * width
    vDSP_vsmul(widthPointer, 1, halfScalarPointer, temporaryWidthPointer, 1, vDSP_Length(elementCount))
    vDSP_vsub(temporaryWidthPointer, 1, centerXPointer, 1, boxesPointer.advanced(by: 1), boxSize, vDSP_Length(elementCount))
    
    //boxes[2](y2) = boxes[0](y1) + height
    vDSP_vadd(boxesPointer, boxSize, heightPointer, boxSize, boxesPointer.advanced(by: 2), 1, vDSP_Length(elementCount))
    
    //boxes[3](x2) = boxes[1](x1) + width
    vDSP_vadd(boxesPointer.advanced(by: 1), boxSize, widthPointer, boxSize, boxesPointer.advanced(by: 3), 1, vDSP_Length(elementCount))
}

func clipBoxes(boxesPointer:UnsafeMutablePointer<Float>,
               elementCount:Int)
{
    var minimum:Float = 0.0
    let minimumPointer = UnsafeMutablePointer<Float>(&minimum)
    var maximum:Float = 1.0
    let maximumPointer = UnsafeMutablePointer<Float>(&maximum)
    
    //Clip between 0 and 1
    vDSP_vclip(boxesPointer, 1, minimumPointer, maximumPointer, boxesPointer, 1, vDSP_Length(elementCount))
}

func nonMaxSupression(boxes:[Float],
                      indices:[Int],
                      iouThreshold:Float,
                      max:Int) -> [Int] {
    var selected:[Int] = []
    
    for index in indices {
        if selected.count >= max { return selected }
        
        let anchorA = CGRect(anchorDatum: boxes[index*4..<index*4+4])
        var shouldSelect = anchorA.width > 0 && anchorA.height > 0
        
        if(shouldSelect) {
            //       Does the current box overlap one of the selected anchors more than the
            //       given threshold amount? Then it's too similar, so don't keep it.
            for j in selected {
                
                let anchorB = CGRect(anchorDatum: boxes[j*4..<j*4+4])
                if IOU(anchorA, anchorB) > iouThreshold {
                    shouldSelect = false
                    break
                }
                
            }
        }
        // This bounding box did not overlap too much with any previously selected
        // bounding box, so we'll keep it.
        if shouldSelect {
            selected.append(index)
        }
    }
    
    return selected
}

//TODO: Vectorize this

extension CGRect
{
    init(anchorDatum:ArraySlice<Float>) {
        let index = anchorDatum.startIndex
        let y1 = CGFloat(anchorDatum[index])
        let x1 = CGFloat(anchorDatum[index+1])
        let y2 = CGFloat(anchorDatum[index+2])
        let x2 = CGFloat(anchorDatum[index+3])
        self = CGRect(x: x1, y:y1, width: x2-x1, height: y2-y1)
    }
}

public func IOU(_ a: CGRect, _ b: CGRect) -> Float {
    let areaA = a.width * a.height
    if areaA <= 0 { return 0 }
    
    let areaB = b.width * b.height
    if areaB <= 0 { return 0 }
    
    let intersectionMinX = max(a.minX, b.minX)
    let intersectionMinY = max(a.minY, b.minY)
    let intersectionMaxX = min(a.maxX, b.maxX)
    let intersectionMaxY = min(a.maxY, b.maxY)
    let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
        max(intersectionMaxX - intersectionMinX, 0)
    return Float(intersectionArea / (areaA + areaB - intersectionArea))
}
