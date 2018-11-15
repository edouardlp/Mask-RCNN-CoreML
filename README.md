# Mask-RCNN (CoreML)

Mask-RCNN using Core ML, Metal 2 and Accelerate (WORK IN PROGRESS). 

This project is a port of https://github.com/matterport/Mask_RCNN for Core ML.

![Example](Example/Screenhots/Screenshot.png)

## Upcoming Features

- CoreML inference of https://github.com/matterport/Mask_RCNN pretrained weights
- Mobile-optimized backbone
- Support for pose estimation

## Goals

Why would you use Mask-RCNN on mobile? The uses cases I see are not real-time (yet). Basically, you would use it in places where accuracy is important, but real-time is not necessarily.
