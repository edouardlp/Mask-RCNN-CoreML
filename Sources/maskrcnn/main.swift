import SwiftCLI

let mainCLI = CLI(name: "maskrcnn")
mainCLI.commands = [ConvertCommand(),EvaluateCommand(),TrainCommand()]
mainCLI.goAndExit()
