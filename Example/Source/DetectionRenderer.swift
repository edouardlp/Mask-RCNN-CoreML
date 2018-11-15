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
    
    static func renderDetections(detections:[Detection],
                                 onImage image:UIImage,
                                 size:CGSize) -> UIImage {
        
        UIGraphicsGetCurrentContext()?.saveGState()
        let colors = [UIColor.red, UIColor.blue, UIColor.green, UIColor.yellow]
        
        var i = -1
        let detectionImages = detections.map { (detection) -> UIImage in
            i += 1
            return renderDetection(detection: detection, inSize: size, color: colors[i%4])
        }
        
        UIGraphicsBeginImageContext(size)
        
        let horizontalScaleFactor = size.width/image.size.width
        let verticalScaleFactor = size.height/image.size.height
        
        let fitsHorizontally = image.size.height*horizontalScaleFactor <= size.height
        
        let scaleFactor = fitsHorizontally ? horizontalScaleFactor : verticalScaleFactor
        
        let imageSize = CGSize(width: image.size.width*scaleFactor, height: image.size.height*scaleFactor)
        
        let horizontalPadding = size.width-imageSize.width
        let verticalPadding = size.height-imageSize.height
        
        image.draw(in: CGRect(origin: CGPoint(x: horizontalPadding/2, y: verticalPadding/2), size: imageSize))
        
        for detectionImage in detectionImages {
            detectionImage.draw(at: CGPoint(x: 0, y: 0))
        }
        
        let outputImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        UIGraphicsGetCurrentContext()?.restoreGState()

        return outputImage ?? image
    }
    
}
