//
//  DetectionRenderer.swift
//  Example
//
//  Created by Edouard Lavery-Plante on 2018-11-14.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import UIKit

class DetectionRenderer {
    
    static func renderMask(mask:CGImage, inSize size:CGSize, color:UIColor) -> UIImage {
        
        UIGraphicsBeginImageContext(size)

        UIGraphicsGetCurrentContext()?.clip(to: CGRect(origin: CGPoint.zero, size: size), mask: mask)
        UIGraphicsGetCurrentContext()?.setFillColor(color.cgColor)
        UIGraphicsGetCurrentContext()?.fill(CGRect(origin: CGPoint.zero, size: size))

        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return UIImage(cgImage: image!.cgImage!, scale: 1.0, orientation: UIImage.Orientation.downMirrored)
    }
    
    static func renderDetection(detection:Detection, inSize size:CGSize, color:UIColor) -> UIImage  {
        
        UIGraphicsBeginImageContext(size)
        
        let transform = CGAffineTransform(scaleX: size.width, y: size.height)
        let scaledBoundingBox = detection.boundingBox.applying(transform)
        
        let path = UIBezierPath(rect:scaledBoundingBox)
        UIGraphicsGetCurrentContext()?.setStrokeColor(color.cgColor)
        path.lineWidth = 3.0
        path.stroke()
        
        if let mask = detection.mask {
            let maskImage = renderMask(mask: mask, inSize: scaledBoundingBox.size, color: color)
            maskImage.draw(at: scaledBoundingBox.origin)
        }
        
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image!
    }
    
    static func renderDetections(detections:[Detection], onImage image:UIImage) -> UIImage {
        UIGraphicsGetCurrentContext()?.saveGState()
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
        UIGraphicsGetCurrentContext()?.restoreGState()

        return outputImage ?? image
    }
    
}
