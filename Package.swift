// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Mask-RCNN-CoreML",
    products: [
        .library(
            name: "Mask-RCNN-CoreML",
            targets: ["Mask-RCNN-CoreML"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-protobuf.git", from: "1.2.0"),
        .package(url: "https://github.com/jakeheis/SwiftCLI", from: "5.2.1"),
    ],
    targets: [
        .target(
            name: "Mask-RCNN-CoreML",
            dependencies: []),
        .target(
            name: "Mask-RCNN-Tool",
            dependencies: ["SwiftProtobuf","SwiftCLI", "Mask-RCNN-CoreML"]),
    ]
)
