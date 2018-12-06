import SwiftCLI

let mainCLI = CLI(name: "Mask-RCNN-Tool")
mainCLI.commands = [ConvertCommand(),EvaluateCommand(),TrainCommand()]
mainCLI.goAndExit()
