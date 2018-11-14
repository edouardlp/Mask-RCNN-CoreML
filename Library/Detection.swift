//
//  Detection.swift
//  Example
//
//  Created by Edouard Lavery-Plante on 2018-11-14.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import Foundation
import CoreGraphics
import CoreML

struct Detection {
    
    let index:Int
    let boundingBox:CGRect
    let classId:Int
    let score:Double
    
    static func detectionsFromFeatureValue(featureValue:MLFeatureValue) -> [Detection] {
        
        guard let rawDetections = featureValue.multiArrayValue else {
            return []
        }
        
        let detectionsCount = Int(truncating: rawDetections.shape[0])
        let detectionStride = Int(truncating: rawDetections.strides[0])
        
        var detections = [Detection]()
        
        for i in 0 ..< detectionsCount {
            
            let score = Double(truncating: rawDetections[i*detectionStride+5])
            if(score > 0.7) {
                
                let classId = Int(truncating: rawDetections[i*detectionStride+4])
                let y1 = CGFloat(truncating: rawDetections[i*detectionStride])
                let x1 = CGFloat(truncating: rawDetections[i*detectionStride+1])
                let y2 = CGFloat(truncating: rawDetections[i*detectionStride+2])
                let x2 = CGFloat(truncating: rawDetections[i*detectionStride+3])
                let width = x2-x1
                let height = y2-y1
                
                let detection = Detection(index:i, boundingBox: CGRect(x: x1, y: y1, width: width, height: height), classId: classId, score: score)
                
                if(width > 0 && height > 0) {
                    detections.append(detection)
                }
                
            }
            
        }
        
        return detections
    }
    
}
