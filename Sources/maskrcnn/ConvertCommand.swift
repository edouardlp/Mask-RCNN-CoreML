import Foundation
import SwiftCLI

class ConvertCommand: Command {
    
    let name = "convert"
    let shortDescription = "Converts trained model to CoreML"
    let modelName = Parameter()
    let configFilePath = Key<String>("--config", description: "Path to config JSON file")
    let weightsFilePath = Key<String>("--weights", description: "Path to HDF5 weights file")
    let outputDirectoryPath = Key<String>("--output_dir", description: "Path to output directory")

    func execute() throws {
        
        guard #available(macOS 10.11, *) else {
            stdout <<< "eval requires macOS >= 10.11"
            return
        }
        
        guard Docker.installed else {
            stdout <<< "Docker is required to run this script."
            return
        }
        
        let name = self.modelName.value
        
        stdout <<< "Converting '\(name)' model to CoreML."
        
        let verbose = false
        
        let currentDirectoryPath = FileManager.default.currentDirectoryPath
        let currentDirectoryURL = URL(fileURLWithPath: currentDirectoryPath)
        
        let buildURL = currentDirectoryURL.appendingPathComponent("Sources/maskrcnn/Python/Conversion")
        let modelURL = currentDirectoryURL.appendingPathComponent(".maskrcnn/models").appendingPathComponent(name)
        let defaultModelDirectoryURL = modelURL.appendingPathComponent("model/")
        let defaultProductsDirectoryURL = modelURL.appendingPathComponent("products/")

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
        
        if let outputDirectoryPath = self.outputDirectoryPath.value {
            let productsURL = URL(fileURLWithPath:outputDirectoryPath, isDirectory:true, relativeTo:currentDirectoryURL)
            mounts.append(Docker.Mount(source:productsURL.standardizedFileURL, destination:dockerProductsDestinationPath))
        } else {
            mounts.append(Docker.Mount(source:defaultProductsDirectoryURL,
                                       destination:dockerProductsDestinationPath))
        }
        
        let docker = Docker(name:"mask-rcnn-convert", buildURL:buildURL)
        
        try docker.build(verbose:verbose)
        try docker.run(mounts:mounts,verbose:verbose)
    }
    
}
