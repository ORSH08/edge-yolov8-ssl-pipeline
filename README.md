# Computer Vision: Semi-Supervised Object Detection Pipeline

**Authors:** Or Shemesh
**Frameworks:** Ultralytics YOLOv8, Microsoft Florence-2, PyTorch  

## Project Overview
This project demonstrates an end-to-end Computer Vision pipeline for object detection, designed with edge-device deployment in mind. We tackle the challenge of manual annotation bottlenecks and natural dataset class imbalances by implementing a **Semi-Supervised Learning (SSL)** pipeline. 

By leveraging a Vision-Language Model (VLM) for zero-shot automated labeling, we curated a highly balanced dataset from raw street photography, which was then used to train and optimize a lightweight YOLOv8 Nano model.

## Key Features & Methodology

### 1. Automated Labeling & Smart Data Ingestion
* **Zero-Shot Detection:** Utilized Microsoft's **Florence-2-large** (running in FP16 precision) to automatically generate bounding boxes for raw images sourced from the Flickr30k dataset.
* **Dynamic Balancing:** Street photography naturally overrepresents the `person` class compared to `vehicle` classes. We built a custom streaming script that dynamically filtered and skipped images to enforce strict class quotas.
* **Result:** Successfully captured an extensive, balanced cache of over **25,100 training** and **5,100 validation** bounding box instances without any manual annotation.

### 2. Edge-Optimized Architecture
* Selected **YOLOv8n (Nano)** for its minimal weight footprint, making it a viable candidate for resource-constrained edge computing environments (e.g., Raspberry Pi, NVIDIA Jetson) where low execution latency and strict memory limits are critical.

### 3. Empirical Augmentation Analysis
We conducted a dual-phase training strategy (30 epochs, 640x640 imgsz) to quantify the impact of data augmentation:
* **Raw Baseline:** Trained with zero augmentations (strictly relying on raw pixel distributions).
* **Augmented Model:** Trained with YOLOv8's standard spatial and color transformations.

## Results & Performance
The comparative analysis provided a powerful demonstration of data augmentation's role in deep learning generalization. The augmented framework outperformed the raw baseline across all key metrics:

| Metric | Raw Baseline | Augmented Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Precision** | 75.3% | **76.7%** | + 1.4% |
| **mAP50-95 (Overall)** | 0.5115 | **0.5191** | + 0.0076 |
| **mAP50 (Vehicle Class)** | 0.610 | **0.627** | + 0.017 |
| **mAP50 (Overall)** | - | **0.732** | Robust edge target |

*Crucially, the synthetic spatial and color variances effectively mitigated the natural dataset imbalance, lifting the minority `vehicle` class mAP50 score significantly. The augmented network learned to map resilient visual features rather than simply memorizing static pixel coordinates.*

## Future Work
* **Test-Time Augmentation (TTA):** Integrating TTA to smooth out localization predictions during active edge video feed testing.
* **Knowledge Distillation:** Implementing a pseudo-labeling loop using a larger foundation network to generate intricate ground-truth coordinates on video feeds, distilling this knowledge directly into our specialized edge model.

## Tech Stack
* **Languages:** Python
* **Deep Learning:** PyTorch, Transformers (Hugging Face), Ultralytics (YOLO)
* **Data Processing:** Pandas, Pillow, OpenCV
* **Visualization:** Matplotlib, TensorBoard

## How to Run
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install transformers==4.46.3 accelerate timm ultralytics