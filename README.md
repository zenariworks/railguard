# Segmentation Comparisson

| Model       | Type                                  | RT | Speed    | Accuracy | Computational Cost | Architecture                 | Application                                    |
|-------------|---------------------------------------|------------|----------|----------|--------------------|--------------------------------|------------------------------------------------|
| [YOLOv8](https://github.com/ultralytics/ultralytics)      | Instance                              | Yes        | Fast     | Moderate | Low                | Anchor-free CNN                | Object detection & instance segmentation       |
| [Detectron2](https://github.com/facebookresearch/detectron2)  | Instance & Semantic                   | No         | Moderate | High     | High               | Mask R-CNN, Faster R-CNN       | Advanced computer vision research              |
| [BiSeNet](https://github.com/CoinCheung/BiSeNet)     | Semantic                              | Yes        | Fast     | Moderate | Low                | Dual-path CNN                  | Autonomous driving, real-time tasks            |
| [DeepLabV3](https://github.com/VainF/DeepLabV3Plus-Pytorch)   | Semantic                              | No         | Moderate | High     | High               | Atrous Convolution             | High-accuracy segmentation tasks               |
| [PSPNet](https://github.com/hszhao/PSPNet)      | Semantic                              | No         | Moderate | High     | High               | Pyramid Pooling CNN            | Large-scale scene parsing                      |
| [FCN](https://github.com/wkentaro/pytorch-fcn)         | Semantic                              | Yes        | Fast     | Moderate | Low                | Fully Convolutional Network    | Simple real-time segmentation tasks            |
| SegNet      | Semantic                              | Yes        | Fast     | Moderate | Low                | Encoder-Decoder                | Efficient segmentation on embedded systems     |
| [UperNet](https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py)     | Panoptic                              | No         | Moderate | High     | High               | Feature Pyramid Network (FPN)  | Autonomous driving, panoptic segmentation      |


### Explanations

* **RT Capable**: Real-Time Capable, indicating whether the model is suitable for real-time applications.
* **Instance**: Instance segmentation focuses on detecting and segmenting each object instance separately.
* **Semantic**: Semantic segmentation classifies each pixel into a class but does not distinguish between individual object instances.
* **Panoptic**: Panoptic segmentation combines both semantic and instance segmentation, segmenting both objects and regions of the image.
* **Speed**: Refers to the general speed of inference and execution.
* **Computational Cost**: Relative computational resources required (e.g., memory, processing power).

### Architectures

* **Anchor-free CNN**: A convolutional neural network that does not use predefined anchor boxes for object detection, making the process more flexible and efficient.
* **Mask R-CNN, Faster R-CNN**: A region-based convolutional neural network that first detects objects (R-CNN) and then generates pixel-wise masks for instance segmentation (Mask R-CNN). Faster R-CNN is an improved version that speeds up the detection process.
* **Dual-path CNN**: A network architecture with two paths—one for capturing spatial detail and the other for global context—used in BiSeNet for real-time segmentation.
* **Atrous Convolution**: Also known as dilated convolution, it expands the receptive field without increasing the number of parameters, helping models like DeepLabV3 capture multi-scale context efficiently.
* **Pyramid Pooling CNN**: Used in PSPNet, this architecture aggregates context from different regions of the image at multiple scales through pyramid pooling to improve segmentation accuracy.
* **Fully Convolutional Network (FCN)**: A simple architecture where all layers are convolutional, enabling dense pixel-wise predictions for segmentation tasks.
* **Encoder-Decoder**: A structure where the encoder reduces the image to a compact feature representation and the decoder upsamples it back to the original resolution, used in SegNet.
* **Feature Pyramid Network (FPN)**: A hierarchical network that uses features at different scales to improve object and scene understanding, used in UperNet for panoptic segmentation.
