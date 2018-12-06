import Foundation
import SwiftCLI

class ConvertCommand: Command {
    
    let name = "convert"
    let shortDescription = "Converts trained model to CoreML"
    let modelName = Parameter()
    
    func execute() throws {
        
        let name = modelName.value
        
        stdout <<< "Converting \(name)!"
        
        let verbose = false
        
        let currentDirectoryPath = FileManager.default.currentDirectoryPath
        let currentDirectoryURL = URL(fileURLWithPath: currentDirectoryPath)
        
        let buildURL = currentDirectoryURL.appendingPathComponent("Sources/Mask-RCNN-Tool/Python/Conversion")
        let modelURL = currentDirectoryURL.appendingPathComponent(".maskrcnn/models").appendingPathComponent(name)
        
        let modelDirectoryURL = modelURL.appendingPathComponent("model/")
        let productsDirectoryURL = modelURL.appendingPathComponent("products/")
        
        let docker = Docker(name:"mask-rcnn-convert", buildURL:buildURL)
        try docker.build(verbose:verbose)
        try docker.run(mounts:[Docker.Mount(source:modelDirectoryURL, destination:"/usr/src/app/model"),
                           Docker.Mount(source:productsDirectoryURL, destination:"/usr/src/app/products")],verbose:verbose)
        
        
        
    }
    
}
