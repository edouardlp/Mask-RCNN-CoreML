import Foundation
import SwiftCLI

class Docker {
    
    class var installed:Bool {
        do {
            let _ = try SwiftCLI.capture("docker",
                                         arguments:["version"])
            return true
        } catch {
            return false
        }
    }
    
    let name:String
    let buildURL:URL
    
    struct Mount {
        let source:URL
        let destination:String
    }
    
    init(name:String,
         buildURL:URL) {
        self.name = name
        self.buildURL = buildURL
    }
    
    func build(verbose:Bool = false) throws {
        let result = try SwiftCLI.capture("docker",
                                          arguments:["build", "-t", self.name, "."],
                                          directory:self.buildURL.relativePath)
        if(verbose) {
            print(result.stdout)
        }
    }
    
    func run(mounts:[Mount], arguments:[String] = [], verbose:Bool = false) throws {
        let uuid = UUID().uuidString
        var allArguments = ["run", "--rm", "--name", uuid]
        for mount in mounts {
            allArguments.append("--mount")
            allArguments.append("type=bind,source=\(mount.source.relativePath),target=\(mount.destination)")
        }
        allArguments.append(self.name)
        allArguments.append(contentsOf:arguments)
        let result = try SwiftCLI.capture("docker", arguments:allArguments)
        if(verbose) {
            print(result.stdout)
        }
    }
    
}
