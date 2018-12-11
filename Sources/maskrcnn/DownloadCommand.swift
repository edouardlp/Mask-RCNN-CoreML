import Foundation
import SwiftCLI
import Alamofire

class DownloadCommand: Command {
    
    let name = "download"
    let shortDescription = "Downloads resources"
    let downloadName = Parameter()
    let version = "0.2"
    
    func execute() throws {
        let name = downloadName.value
        stdout <<< "Downloading \(name) resources. This may take a while..."
        self.download(files:["anchors.bin","MaskRCNN.mlmodel","Mask.mlmodel","Classifier.mlmodel"])
        stdout <<< "Done."
    }
    
    func download(files:[String]){
        let queue = DispatchQueue.global(qos: .userInitiated)
        let group = DispatchGroup()
        for file in files {
            download(file:file, group:group, queue:queue)
        }
        group.wait()
    }
    
    func download(file:String,
                  group:DispatchGroup,
                  queue:DispatchQueue) {
        
        let urlString = "https://github.com/edouardlp/Mask-RCNN-CoreML/releases/download/\(self.version)/\(file)"
        
        let currentDirectoryPath = FileManager.default.currentDirectoryPath
        let currentDirectoryURL = URL(fileURLWithPath: currentDirectoryPath)
        
        let modelURL = currentDirectoryURL.appendingPathComponent(".maskrcnn/models").appendingPathComponent("coco")
        let defaultProductsDirectoryURL = modelURL.appendingPathComponent("products/")
        
        let destination: DownloadRequest.DownloadFileDestination = { _, _ in
            let fileURL = defaultProductsDirectoryURL.appendingPathComponent(file)
            return (fileURL, [.removePreviousFile, .createIntermediateDirectories])
        }
        group.enter()
        Alamofire.download(urlString, to: destination).response(queue:queue) {
            _ in
            group.leave()
        }
    }
    
}

