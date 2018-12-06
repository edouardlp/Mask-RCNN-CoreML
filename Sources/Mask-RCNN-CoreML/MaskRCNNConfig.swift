//
//  MaskRCNNConfig.swift
//  Example
//
//  Created by Edouard Lavery-Plante on 2018-12-07.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//
import Foundation

public class MaskRCNNConfig {
    
    public static let defaultConfig = MaskRCNNConfig()
    
    //TODO: generate the anchors on demand based on image shape, this will save 5mb
    public var anchorsURL:URL?
    public var compiledClassifierModelURL:URL?
    public var compiledMaskModelURL:URL?
    
}
