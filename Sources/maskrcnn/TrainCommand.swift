import Foundation
import SwiftCLI

class TrainCommand: Command {
    
    let name = "train"
    let shortDescription = "Trains model"
    let modelName = Parameter()
    
    func execute() throws {
        let name = modelName.value
        stdout <<< "Coming soon \(name)!."
    }
    
}

