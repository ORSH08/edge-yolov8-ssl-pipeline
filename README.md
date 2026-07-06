# Semi-Supervised Object Detection for Edge Devices
### Teacher (Florence-2) → Student (YOLOv8n), built on the Flickr30k image pool

**Course:** 10224 Introduction to Computer Vision — Final Project 2026  
**Authors:** Or Shemesh

---

## Project Overview
This project presents an end-to-end Computer Vision pipeline for object detection, specifically designed for deployment on computationally constrained edge devices (e.g., Raspberry Pi, NVIDIA Jetson). To overcome the bottleneck of manual annotation and natural dataset class imbalance, we implemented a **Semi-Supervised Learning** architecture using a Teacher-Student paradigm.

The large Vision-Language Model (VLM) **Florence-2-large** (Teacher network) was utilized for zero-shot automated pseudo-label generation. These generated annotations formed a controlled, filtered dataset used to train and distill a lightweight **YOLOv8 Nano** model (Student network) capable of real-time inference on edge hardware.

### Detection Classes:
* **Class 0 - Person:** Includes labels for *person, man, woman, boy, girl, child, people*.
* **Class 1 - Vehicle:** Includes labels for *car, truck, bus, motorcycle, motorbike, bike, bicycle*.

---

## Key Engineering Solutions
The current iteration of our pipeline addresses three major architectural flaws present in earlier versions:

1. **Extreme GPU Optimization (Smart Filtering):** Previously, the heavy Florence-2 model was forced to process all ~31,000 images in the Flickr30k dataset. We introduced a text-based Caption Filter that analyzes the image description first, sending only relevant images to the Teacher model. Coupled with a "Vehicle-first queue" to prioritize the minority class and a hard-stop quota mechanism, **this architectural shift saved ~88% of the Teacher's processing workload** (processing only ~3,700 images).
2. **Empirical Validation (Baseline Metrics):** To prevent unsubstantiated claims of improvement, the pipeline first evaluates the accuracy of a raw, untrained YOLOv8n model (Untrained Baseline). From there, the system plots a precise comparative improvement graph across epochs to visualize actual learning progress.
3. **Full Reproducibility (Central Configuration):** All experimental and training hyperparameters (especially Learning Rate and Augmentation parameters) are centralized into a single configuration block (`COMMON_TRAIN`, `AUG_OFF`, `AUG_ON`). These are explicitly passed to the training commands and read back from disk upon completion to ensure full result reproducibility.

---

## Degrees of Freedom & Geometric Filtering
Since the Florence-2 <OD> task does not return confidence scores, label quality is enforced through strict geometric filters:
* **Minimum Box Area:** Set to 0.3% of the image area to filter out background noise and microscopic specks.
* **Maximum Box Area:** Capped at 90% of the image area to reject degenerate boxes that engulf the entire frame.
* **Class Quotas:** Hard code assertions ensure at least 500 training images and 100 validation images per class before the pipeline is allowed to proceed to the training phase.

---

## Architecture Selection & Dismissals
In accordance with the project requirements, we evaluated multiple model architectures for our edge-deployment objective:

* **YOLOv8n (Chosen):** A modern, anchor-free one-stage detector (Nano version). It offers the optimal balance between high precision and a near-zero memory footprint (3.2M parameters, ~6MB file size). It supports one-line export to edge-accelerated formats (ONNX, TFLite), making it the perfect fit for Raspberry Pi deployment.
* **Faster R-CNN (Dismissed):** A classic two-stage detector. While highly accurate, it is exceptionally computationally heavy. Its latency is simply too high to achieve real-time performance on weak edge hardware without a dedicated GPU.
* **RetinaNet (Dismissed):** A solid one-stage detector utilizing Focal Loss, but it typically relies on large backbones (e.g., ResNet-50). It is less efficient than YOLOv8n for a simple two-class problem and lacks the mature out-of-the-box edge tooling provided by Ultralytics.
* **EfficientDet (Dismissed):** Highly efficient on paper, but practically, its training setup and edge-export tooling are less stable and more complex compared to the Ultralytics ecosystem.
* **MobileNetV4 (Dismissed):** While genuinely edge-friendly, the assignment explicitly marks this as "advanced work" requiring manual integration of a custom detection head onto the backbone. Given project deadlines, this integration posed an unnecessary risk, especially when YOLOv8n provides similar edge benefits out-of-the-box.

---

## Results & Performance Metrics
Training the model with spatial and color augmentations significantly improved the network's generalization capabilities compared to the raw baseline, particularly in handling the inherent imbalance of the vehicle class:

| Metric | Raw Baseline | Augmented Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Precision** | 75.3% | **76.7%** | + 1.4% |
| **mAP50 (Vehicle Class)** | 0.610 | **0.627** | + 0.017 |
| **mAP50 (Overall)** | 0.600 | **0.732** | **+ 0.132** |
| **mAP50-95 (Overall)** | 0.5115 | **0.5191** | + 0.0076 |

*Note: Synthetic augmentations prevented the model from merely memorizing static pixel coordinates, forcing it to learn resilient visual features.*

---

## Future Work
1. **Test-Time Augmentation (TTA):** Implement augmentation checks during inference to stabilize and refine detection boxes on live video streams at the edge.
2. **Ensemble Verification:** Utilize an ensemble of different Teacher models during the dataset generation phase to cross-reference predictions. This will heavily filter out pseudo-label noise, which is currently the primary source of error in the pipeline.
3. **Hardware Benchmarking:** Conduct physical measurements of inference speed, memory utilization, and power consumption of the exported `best.pt` model running natively on a Jetson or Raspberry Pi device.