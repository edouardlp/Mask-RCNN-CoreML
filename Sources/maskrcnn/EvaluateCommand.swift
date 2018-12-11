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
    let evalDataset = Parameter()
    let configFilePath = Key<String>("--config", description: "Path to config JSON file")
    let weightsFilePath = Key<String>("--weights", description: "Path to HDF5 weights file")
    let productsDirectoryPath = Key<String>("--products_dir", description: "Path to products directory")
    let yearOption = Key<String>("--year", description: "COCO dataset year")
    let typeOption = Key<String>("--type", description: "COCO dataset type")
    let compareFlag = Flag("-c", "--compare")

    func execute() throws {

        guard #available(macOS 10.14, *) else {
            stdout <<< "eval requires macOS >= 10.14"
            return
        }
        
        guard Docker.installed else {
            stdout <<< "Docker is required to run this script."
            return
        }
        
        let name = self.modelName.value
        let evalDataset = self.evalDataset.value
        
        stdout <<< "Evaluating \(name) using \(evalDataset)"
        
        let currentDirectoryPath = FileManager.default.currentDirectoryPath
        let currentDirectoryURL = URL(fileURLWithPath: currentDirectoryPath)
        
        let buildURL = currentDirectoryURL.appendingPathComponent("Sources/maskrcnn/Python/COCOEval")
        
        let verbose = true
        let docker = Docker(name:"mask-rcnn-evaluate", buildURL:buildURL)
        try docker.build(verbose:verbose)
        
        let defaultModelsURL = currentDirectoryURL.appendingPathComponent(".maskrcnn/models").appendingPathComponent(name)
        let defaultDataURL = currentDirectoryURL.appendingPathComponent(".maskrcnn/data")
        let defaultModelDirectoryURL = defaultModelsURL.appendingPathComponent("model")

        let productsURL:URL = {
            () -> URL in
            guard let productsDirectoryPath = productsDirectoryPath.value else {
                return defaultModelsURL.appendingPathComponent("products")
            }
            return URL(fileURLWithPath:productsDirectoryPath, isDirectory:false, relativeTo:currentDirectoryURL).standardizedFileURL
        }()

        let mainModelURL = productsURL.appendingPathComponent("MaskRCNN.mlmodel")
        let classifierModelURL = productsURL.appendingPathComponent("Classifier.mlmodel")
        let maskModelURL = productsURL.appendingPathComponent("Mask.mlmodel")
        let anchorsURL = productsURL.appendingPathComponent("anchors.bin")
        
        let cocoURL = defaultDataURL.appendingPathComponent("coco_eval")
        let annotationsDirectoryURL = cocoURL
        
        let year = yearOption.value ?? "2017"
        let type = typeOption.value ?? "val"
        let imagesDirectoryURL = cocoURL.appendingPathComponent("\(type)\(year)")
        
        var mounts = [Docker.Mount]()
        
        let dockerConfigDestinationPath = "/usr/src/app/model/config.json"
        
        if let configFilePath = self.configFilePath.value {
            let configURL = URL(fileURLWithPath:configFilePath, isDirectory:false, relativeTo:currentDirectoryURL)
            mounts.append(Docker.Mount(source:configURL.standardizedFileURL, destination:dockerConfigDestinationPath))
        } else {
            mounts.append(Docker.Mount(source:defaultModelDirectoryURL.appendingPathComponent("config.json"),
                                       destination:dockerConfigDestinationPath))
        }
        
        let dockerWeightsDestinationPath = "/usr/src/app/model/weights.h5"
        
        if let weightsFilePath = self.weightsFilePath.value {
            let weightsURL = URL(fileURLWithPath:weightsFilePath, isDirectory:false, relativeTo:currentDirectoryURL)
            mounts.append(Docker.Mount(source:weightsURL.standardizedFileURL, destination:dockerWeightsDestinationPath))
        } else {
            mounts.append(Docker.Mount(source:defaultModelDirectoryURL.appendingPathComponent("weights.h5"),
                                       destination:dockerWeightsDestinationPath))
        }
        
        let dockerProductsDestinationPath = "/usr/src/app/products"
        mounts.append(Docker.Mount(source:productsURL,
                                   destination:dockerProductsDestinationPath))
        
        let dockerCocoDataDestinationPath = "/usr/src/app/data/coco"
        mounts.append(Docker.Mount(source:cocoURL,
                                   destination:dockerCocoDataDestinationPath))
        
        let temporaryDirectory = currentDirectoryURL.appendingPathComponent(".maskrcnn/tmp")
        try Foundation.FileManager.default.createDirectory(at: temporaryDirectory, withIntermediateDirectories: true, attributes: nil)
        let fileName = NSUUID().uuidString+".pb"
        let outputDataURL = temporaryDirectory.appendingPathComponent(fileName)
        
        let output = try evaluate(datasetId:evalDataset,
                                     modelURL:mainModelURL,
                                     classifierModelURL:classifierModelURL,
                                     maskModelURL:maskModelURL,
                                     anchorsURL:anchorsURL,
                                     annotationsDirectoryURL:annotationsDirectoryURL,
                                     imagesDirectoryURL:imagesDirectoryURL,
                                     year:year,
                                     type:type)
        let outputData = try output.serializedData()
        try outputData.write(to:outputDataURL)
        
        let dockerResultsDestinationPath = "/usr/src/app/results"
        mounts.append(Docker.Mount(source:outputDataURL.standardizedFileURL,
                                   destination:dockerResultsDestinationPath))
        var arguments = ["--coco_year", year, "--coco_type", type, "--results_path", "results"]
        if(compareFlag.value){
            arguments.append("--compare")
            arguments.append("--compare")
        }
        try docker.run(mounts:mounts, arguments:arguments, verbose:verbose)
        try Foundation.FileManager.default.removeItem(at:outputDataURL)
    }
}

@available(macOS 10.14, *)
func evaluate(datasetId:String,
              modelURL:URL,
              classifierModelURL:URL,
              maskModelURL:URL,
              anchorsURL:URL,
              annotationsDirectoryURL:URL,
              imagesDirectoryURL:URL,
              year:String,
              type:String) throws -> Results {
    
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
    
    var outputResults = [Result]()
    
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
                throw "Error during prediction"
        }
        let end = Date().timeIntervalSinceReferenceDate
        let detections = detectionsFromFeatureValue(featureValue: detectionsFeatureValue, maskFeatureValue:maskFeatureValue)
        
        let result = Result.with {
            let image = Result.ImageInfo.with {
                $0.id = String(image.id)
                $0.datasetID = datasetId
                $0.width = Int32(image.width)
                $0.height = Int32(image.height)
            }
            $0.imageInfo = image
            $0.detections = detections
        }
        outputResults.append(result)
        print(end-start)
    }
    
    let output = Results.with {
        $0.results = outputResults
    }
    return output
}

@available(macOS 10.14, *)
func detectionsFromFeatureValue(featureValue: MLFeatureValue,
                                maskFeatureValue:MLFeatureValue) -> [Result.Detection] {
    
    guard let rawDetections = featureValue.multiArrayValue else {
        return []
    }
    
    let detectionsCount = Int(truncating: rawDetections.shape[0])
    let detectionStride = Int(truncating: rawDetections.strides[0])
    var detections = [Result.Detection]()

    for i in 0 ..< detectionsCount {
        
        let probability = Double(truncating: rawDetections[i*detectionStride+5])
        if(probability > 0.7) {
            
            let classId = Int(truncating: rawDetections[i*detectionStride+4])
            let classLabel = "test"
            let y1 = Double(truncating: rawDetections[i*detectionStride])
            let x1 = Double(truncating: rawDetections[i*detectionStride+1])
            let y2 = Double(truncating: rawDetections[i*detectionStride+2])
            let x2 = Double(truncating: rawDetections[i*detectionStride+3])
            let width = x2-x1
            let height = y2-y1
            
            let detection =  Result.Detection.with {
                $0.probability = probability
                $0.classID = Int32(classId)
                $0.classLabel = classLabel
                $0.boundingBox = Result.Rect.with {
                    $0.origin = Result.Origin.with {
                        $0.x = x1
                        $0.y = y1
                    }
                    $0.size = Result.Size.with {
                        $0.width = width
                        $0.height = height
                    }
                }
            }
            detections.append(detection)
        }
        
    }
    return detections
}
