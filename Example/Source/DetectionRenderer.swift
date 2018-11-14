//
//  DetectionRenderer.swift
//  Example
//
//  Created by Edouard Lavery-Plante on 2018-11-14.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import UIKit

class DetectionRenderer {
    
    static func renderDetection(detection:Detection, inSize size:CGSize, color:UIColor) -> UIImage  {
        
        UIGraphicsBeginImageContext(size)
        
        let transform = CGAffineTransform(scaleX: size.width, y: size.height)
        let scaledBoundingBox = detection.boundingBox.applying(transform)
        
        let path = UIBezierPath(rect:scaledBoundingBox)
        UIGraphicsGetCurrentContext()?.setStrokeColor(color.cgColor)
        path.lineWidth = 3.0
        path.stroke()
        
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image!
    }
    
    static func renderDetections(detections:[Detection], onImage image:UIImage) -> UIImage {
        
        let colors = [UIColor.red, UIColor.blue, UIColor.green, UIColor.yellow]
        
        var i = -1
        let detectionImages = detections.map { (detection) -> UIImage in
            i += 1
            return renderDetection(detection: detection, inSize: image.size, color: colors[i%4])
        }
        
        UIGraphicsBeginImageContext(image.size)
        
        image.draw(at: CGPoint(x: 0, y: 0))
        
        for detectionImage in detectionImages {
            detectionImage.draw(at: CGPoint(x: 0, y: 0))
        }
        
        let outputImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return outputImage ?? image
    }
    
}
