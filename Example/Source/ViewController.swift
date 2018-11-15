//
//  ViewController.swift
//  Example
//
//  Created by Edouard Lavery-Plante on 2018-11-14.
//  Copyright © 2018 Edouard Lavery Plante. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var selectPhotoItem:UIBarButtonItem!
    @IBOutlet weak var imageView:UIImageView!
    @IBOutlet weak var label:UILabel!
    @IBOutlet weak var activityIndicator:UIActivityIndicatorView!
    
    private enum State {
        case initial
        case calculating
        case displayingResult
    }
    
    private var state:State = .initial {
        didSet {
            self.handleChangedState(state: self.state)
        }
    }
    
    private var selectedImage:UIImage? = nil
    
    private lazy var maskRCNNRequest: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: MaskRCNN().model)
            let request = VNCoreMLRequest(model: model, completionHandler: {
                [weak self] request, error in
                self?.processMaskRCNNRequest(for: request, error: error)
            })
            request.imageCropAndScaleOption = .scaleFit
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        self.title = "Mask-RCNN Example"
    }
    
    //MARK: Actions
    
    @IBAction func selectPhoto(sender:Any?) {
        
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            self.presentPicker(sourceType: .photoLibrary)
            return
        }
        
        let alertController = UIAlertController(title: nil, message: nil, preferredStyle: .actionSheet)
        
        let cameraAction = UIAlertAction(title: "Take a Photo", style: .default) {
            [weak self](_) in
            self?.presentPicker(sourceType: .camera)
        }
        alertController.addAction(cameraAction)

        let libraryAction = UIAlertAction(title: "Choose from Library", style: .default) {
            [weak self](_) in
            self?.presentPicker(sourceType: .photoLibrary)
        }
        alertController.addAction(libraryAction)
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
        alertController.addAction(cancelAction)

        self.present(alertController, animated: true, completion: nil)
    }
    
    //MARK: UIImagePickerControllerDelegate
    
    @objc func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        guard state != .calculating else {
            return
        }
        
        picker.dismiss(animated: true, completion: nil)
        
        self.state = .calculating

        guard let image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage else {
            return
        }
        
        self.selectedImage = image
        
        let orientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue))!
        guard let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                try handler.perform([self.maskRCNNRequest])
            } catch {
                DispatchQueue.main.async {
                    print("Failed to perform Mask-RCNN.\n\(error.localizedDescription)")
                    self.handleFailure()
                }
            }
        }
    }
    
    //MARK: Private Methods
    
    private func presentPicker(sourceType:UIImagePickerController.SourceType) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = sourceType
        self.present(picker, animated: true, completion: nil)
    }
    
    private func handleChangedState(state:State) {
        switch(state) {
        case .initial:
            self.imageView.image = nil
            self.activityIndicator.stopAnimating()
            self.label.text = "Choose a Photo"
            self.label.isHidden = false
            self.selectPhotoItem.isEnabled = true
        case .calculating:
            self.imageView.image = nil
            self.activityIndicator.startAnimating()
            self.label.text = "Calculating..."
            self.label.isHidden = false
            self.selectPhotoItem.isEnabled = false
        case .displayingResult:
            self.activityIndicator.stopAnimating()
            self.label.isHidden = true
            self.selectPhotoItem.isEnabled = true
        }
    }
    
    private func processMaskRCNNRequest(for request: VNRequest, error: Error?) {
        
        guard let selectedImage = self.selectedImage,
            let results = request.results as? [VNCoreMLFeatureValueObservation],
            let detectionsFeatureValue = results.first?.featureValue,
            let maskFeatureValue = results.last?.featureValue else {
            DispatchQueue.main.async {
                print("Failed to perform Mask-RCNN.\n\(error?.localizedDescription ?? "")")
                self.handleFailure()
            }
            return
        }
        
        let detections = Detection.detectionsFromFeatureValue(featureValue: detectionsFeatureValue, maskFeatureValue:maskFeatureValue)
        
        print(detections)
        
        let resultImage = DetectionRenderer.renderDetections(detections: detections, onImage: selectedImage)
        
        DispatchQueue.main.async {
            self.handleSuccess(image: resultImage)
        }
    }
    
    private func handleSuccess(image:UIImage) {
        self.state = .displayingResult
        self.selectedImage = nil
        self.imageView.image = image
    }
    
    private func handleFailure() {
        
        self.state = .initial
        
        let alertController = UIAlertController(title: "Error", message: "An error occurred attempting to run Mask-RCNN", preferredStyle: .alert)
        
        let cancelAction = UIAlertAction(title: "Ok", style: .cancel, handler: nil)
        alertController.addAction(cancelAction)
        
        self.present(alertController, animated: true, completion: nil)
    }

}

