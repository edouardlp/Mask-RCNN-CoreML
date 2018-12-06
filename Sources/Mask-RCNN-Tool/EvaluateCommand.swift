import Foundation
import Vision
import CoreML
import QuartzCore

import SwiftCLI
import Mask_RCNN_CoreML

class EvaluateCommand: Command {
    
    let name = "evaluate"
    let shortDescription = "Evaluates CoreML model against validation data"
    let modelName = Parameter()
    let evalType = Parameter()

    func execute() throws {

        guard #available(macOS 10.14, *) else {
            print("eval requires macOS >= 10.13")
            return
        }
        
        let name = modelName.value
        let evalType = modelName.value
        
        stdout <<< "Evaluating \(name) using \(evalType)"
        
        let currentDirectoryPath = FileManager.default.currentDirectoryPath
        let currentDirectoryURL = URL(fileURLWithPath: currentDirectoryPath)
        let modelURL = currentDirectoryURL.appendingPathComponent(".maskrcnn/models").appendingPathComponent(name)
        let dataURL = currentDirectoryURL.appendingPathComponent(".maskrcnn/data")
        
        let productsURL = modelURL.appendingPathComponent("products")

        let mainModelURL = productsURL.appendingPathComponent("MaskRCNN.mlmodel")
        let classifierModelURL = productsURL.appendingPathComponent("Classifier.mlmodel")
        let maskModelURL = productsURL.appendingPathComponent("Mask.mlmodel")
        let anchorsURL = productsURL.appendingPathComponent("anchors.bin")
        
        let cocoURL = dataURL.appendingPathComponent("coco_eval")
        let annotationsDirectoryURL = cocoURL
        let imagesDirectoryURL = cocoURL.appendingPathComponent("val2017")
        
        let year = "2017"
        let type = "val"
        
        try evaluate(modelURL:mainModelURL,
                     classifierModelURL:classifierModelURL,
                     maskModelURL:maskModelURL,
                     anchorsURL:anchorsURL,
                     annotationsDirectoryURL:annotationsDirectoryURL,
                     imagesDirectoryURL:imagesDirectoryURL,
                     year:year,
                     type:type)
        
    }
}

@available(macOS 10.14, *)
func evaluate(modelURL:URL,
              classifierModelURL:URL,
              maskModelURL:URL,
              anchorsURL:URL,
              annotationsDirectoryURL:URL,
              imagesDirectoryURL:URL,
              year:String,
              type:String) throws {
    
    MaskRCNNConfig.defaultConfig.anchorsURL = anchorsURL
    
    let compiledClassifierUrl = try MLModel.compileModel(at: classifierModelURL)
    MaskRCNNConfig.defaultConfig.compiledClassifierModelURL = compiledClassifierUrl
    
    let compiledMaskUrl = try MLModel.compileModel(at: maskModelURL)
    MaskRCNNConfig.defaultConfig.compiledMaskModelURL = compiledMaskUrl
    
    let compiledUrl = try MLModel.compileModel(at: modelURL)
    let model = try MLModel(contentsOf: compiledUrl)
    
    let vnModel = try VNCoreMLModel(for:model)
    let request = VNCoreMLRequest(model: vnModel)
    request.imageCropAndScaleOption = .scaleFit
    
    let instancesURL = annotationsDirectoryURL.appendingPathComponent("instances_\(type)\(year).json")
    
    let coco = try COCO(url:instancesURL)
    
    var iterator = coco.makeImageIterator(limit:5, sortById:true)
    while let item = iterator.next() {
        let start = Date().timeIntervalSinceReferenceDate
        let image = item.0
        let imageURL = imagesDirectoryURL.appendingPathComponent(image.fileName)
        let ciImage = CIImage(contentsOf:imageURL)!
        let handler = VNImageRequestHandler(ciImage: ciImage)
        try handler.perform([request])
        
        guard let results = request.results as? [VNCoreMLFeatureValueObservation],
            let detectionsFeatureValue = results.first?.featureValue,
            let maskFeatureValue = results.last?.featureValue else {
                return
        }
        let end = Date().timeIntervalSinceReferenceDate
        let detections = Detection.detectionsFromFeatureValue(featureValue: detectionsFeatureValue, maskFeatureValue:maskFeatureValue)
        print(detections.count)
        print(end-start)
    }
}
