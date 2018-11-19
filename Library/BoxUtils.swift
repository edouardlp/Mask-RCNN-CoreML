//
//  BoxUtils.swift
//  Mask-RCNN-CoreML
//
//  Created by Edouard Lavery-Plante on 2018-11-16.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import Accelerate

typealias BoxArray = [Float]

extension Array where Element == Float {
    mutating func boxReference() -> BoxReference {
        return BoxReference(pointer: UnsafeMutablePointer(&self), boxCount: self.count/4)
    }
}

struct BoxReference {
    
    let pointer:UnsafeMutablePointer<Float>
    let boxCount:Int
    
    var count:Int {
        return boxCount * 4
    }
}

extension BoxReference {
    
    func applyBoxDeltas(_ deltas:BoxArray){
        
        precondition(self.count == deltas.count)
        
        for i in 0 ..< self.boxCount {
            
            let boxPointer = self.pointer.advanced(by: i*4)
            
            let y1 = boxPointer.pointee
            let x1 = boxPointer.advanced(by: 1).pointee
            let y2 = boxPointer.advanced(by: 2).pointee
            let x2 = boxPointer.advanced(by: 3).pointee
            
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
            
            boxPointer.assign(repeating: resultY1, count: 1)
            boxPointer.advanced(by: 1).assign(repeating: resultX1, count: 1)
            boxPointer.advanced(by: 2).assign(repeating: resultY2, count: 1)
            boxPointer.advanced(by: 3).assign(repeating: resultX2, count: 1)
        }
    }
    
    func clip() {
        var minimum:Float = 0.0
        let minimumPointer = UnsafeMutablePointer<Float>(&minimum)
        var maximum:Float = 1.0
        let maximumPointer = UnsafeMutablePointer<Float>(&maximum)
        //Clip all balues between 0 and 1
        vDSP_vclip(self.pointer, 1, minimumPointer, maximumPointer, self.pointer, 1, vDSP_Length(self.count))
    }
    
}
