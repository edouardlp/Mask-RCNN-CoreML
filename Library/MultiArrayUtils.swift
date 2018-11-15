//
//  MultiArrayUtils.swift
//  Example
//
//  Created by Edouard Lavery-Plante on 2018-11-15.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML

extension MLMultiArray {
    
    func padTailWithZeros(startIndex:Int, count:Int) {
        
        guard count > 0 else {
            return
        }
        
        var paddingBuffer = Array<Float>(repeating: 0.0, count: count)
        let paddingBufferPointer = UnsafeMutableRawPointer(&paddingBuffer)
        self.dataPointer.advanced(by: startIndex*MemoryLayout<Float>.size).copyMemory(from: paddingBufferPointer, byteCount: count*MemoryLayout<Float>.size)
    }
    
}
