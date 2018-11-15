# Mask-RCNN (CoreML)

Mask-RCNN using Core ML, Metal 2 and Accelerate.

<div align="center">
<img src="https://github.com/edouardlp/Mask-RCNN-CoreML/blob/master/Example/Screenshots/Screenshot1.png" alt="Example" width="800" height="599" />
</div>

## Mask-RCNN

Mask-RCNN is a general framework for object instance segmentation. It detects objects, the class they belong to, their bounding box and segmentation masks.

## Motivation

Mask-RCNN is not fast, especially with the current ResNet101 + FPN backbone.

There are much faster models for object detection such as SSDLite and YOLO.

This model will only be useful if instance segmentation is valuable for your use-case.

## Examples

![Example 1](Example/Screenshots/Screenshot2.png)
![Example 2](Example/Screenshots/Screenshot3.png)
![Example 3](Example/Screenshots/Screenshot4.png)

## Requirements

- Xcode 10.1
- iOS 12
- (More requirements details coming soon)

## Usage



## Roadmap

- Inference for all weights generated with https://github.com/matterport/Mask_RCNN. (Support for all configurations)
- Mobile-optimized backbone
- Support for pose estimation

## Author

Édouard Lavery-Plante, ed@laveryplante.com

## Credits

- [Original Paper](https://arxiv.org/abs/1703.06870)
- [Matterport Implementation](https://github.com/matterport/Mask_RCNN)
- [Inspiration](http://machinethink.net/blog//)

## References
- [Vision Framework](https://developer.apple.com/documentation/vision)
- [CoreML Framework](https://developer.apple.com/documentation/coreml)
- [coremltools](https://pypi.python.org/pypi/coremltools)
