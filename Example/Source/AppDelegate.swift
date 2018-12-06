//
//  AppDelegate.swift
//  Example
//
//  Created by Edouard Lavery-Plante on 2018-11-14.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        
        MaskRCNNConfig.defaultConfig.anchorsURL = Bundle.main.url(forResource: "anchors", withExtension: "bin")!
        MaskRCNNConfig.defaultConfig.compiledClassifierModelURL = Bundle.main.url(forResource: "Classifier", withExtension:"mlmodelc")!
        MaskRCNNConfig.defaultConfig.compiledMaskModelURL = Bundle.main.url(forResource: "Mask", withExtension:"mlmodelc")!
        
        return true
    }
}

