//
//  Utils.swift
//  Mask-RCNN-Demo
//
//  Created by Edouard Lavery-Plante on 2018-11-06.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import Accelerate
import CoreML

extension UnsafePointer where Pointee == Float {
    
    func stridedSlice(begin:Int, count:Int, stride:Int) -> [Float] {
        let dataPointer = self.advanced(by: begin)
        var result = Array<Float>(repeating:0.0, count: count)
        let resultPointer = UnsafeMutablePointer<Float>(&result)
        cblas_scopy(Int32(count), dataPointer, Int32(stride), resultPointer, 1)
        return result
    }
    
    func indexed(indices:[Float]) -> [Float] {
        var results = Array<Float>(repeating: 0, count: indices.count)
        var indices = indices
        vDSP_vindex(self,
                    UnsafeMutablePointer<Float>(&indices),
                    1,
                    UnsafeMutablePointer<Float>(&results),
                    1,
                    vDSP_Length(indices.count))
        return results
    }
}

extension Array where Element == Float {
    
    func sortedIndices(ascending:Bool) -> [UInt] {
        var array = self
        let totalElements = vDSP_Length(array.count)
        var indices = Array<vDSP_Length>(0 ..< totalElements)
        vDSP_vsorti(UnsafeMutablePointer<Float>(&array),
                    UnsafeMutablePointer<vDSP_Length>(&indices),
                    nil,
                    vDSP_Length(array.count),
                    ascending ? 1 : -1)
        return indices
    }
    
}

extension ArraySlice where Element == UInt {
    
    func toFloat() -> [Float] {
        return Array<Float>(self.map({ (integerValue) -> Float in
            return Float(integerValue)
        }))
    }
    
}

extension MLMultiArray {
    
    func floatDataPointer() -> UnsafePointer<Float> {
        assert(self.dataType == .float32)
        let dataPointer = UnsafePointer<Float>(OpaquePointer(self.dataPointer))
        return dataPointer
    }
    
    func padTailWithZeros(startIndex:Int, count:Int) {
        
        guard count > 0 else {
            return
        }
        
        var paddingBuffer = Array<Float>(repeating: 0.0, count: count)
        let paddingBufferPointer = UnsafeMutableRawPointer(&paddingBuffer)
        self.dataPointer.advanced(by: startIndex*MemoryLayout<Float>.size).copyMemory(from: paddingBufferPointer, byteCount: count*MemoryLayout<Float>.size)
    }
    
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

func elementWiseMultiply(matrixPointer:UnsafeMutablePointer<Float>,
                         vectorPointer:UnsafeMutablePointer<Float>,
                         height:Int,
                         width:Int) {
    for i in 0 ..< width {
        vDSP_vsmul(matrixPointer.advanced(by: i), width, vectorPointer.advanced(by: i), matrixPointer.advanced(by: i), width, vDSP_Length(height))
    }
}

//nonMaxSupression Adapted from https://github.com/hollance/CoreMLHelpers

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
