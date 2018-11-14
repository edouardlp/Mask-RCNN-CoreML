//
//  DebugLayer.swift
//  Mask-RCNN-Demo
//
//  Created by Edouard Lavery-Plante on 2018-11-09.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreML

@objc(DebugLayer) class DebugLayer: NSObject, MLCustomLayer {
    
    
    required init(parameters: [String : Any]) throws {
        super.init()
    }
    
    func setWeightData(_ weights: [Data]) throws {
        //No-op
    }
    
    func outputShapes(forInputShapes inputShapes: [[NSNumber]]) throws -> [[NSNumber]] {
        return inputShapes
    }
    
    func evaluate(inputs: [MLMultiArray], outputs: [MLMultiArray]) throws {
        for (i,input) in inputs.enumerated() {
            let inputPointer = input.dataPointer
            outputs[i].dataPointer.copyMemory(from: inputPointer, byteCount: input.count*4)
        }
    }
    
    
}
