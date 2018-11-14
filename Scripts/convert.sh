#!/bin/sh
docker build -t mask_rcnn_coreml_converter Conversion/
docker run -it \
--rm \
--name mask_rcnn_coreml_convert \
--mount type=bind,source="$(pwd)"/Data,target=/usr/src/app/Data \
mask_rcnn_coreml_converter
