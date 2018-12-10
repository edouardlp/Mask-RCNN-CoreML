import Foundation
import SwiftCLI

class Docker {
    
    class var installed:Bool {
        do {
            try SwiftCLI.run("docker",
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
        try SwiftCLI.run("docker",
                         arguments:["build", "-t", self.name, "."],
                         directory:self.buildURL.relativePath)
    }
    
    func run(mounts:[Mount], verbose:Bool = false) throws {
        let uuid = UUID().uuidString
        var arguments = ["run", "--rm", "--name", uuid]
        for mount in mounts {
            arguments.append("--mount")
            arguments.append("type=bind,source=\(mount.source.relativePath),target=\(mount.destination)")
        }
        arguments.append(self.name)
        print(arguments)
        try SwiftCLI.run("docker",
                         arguments:arguments)
    }
    
}
