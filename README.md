# Car-Plate-Recognition-and-Reconstruction-with-Deep-Learning

Automatic car plate recognition is a crucial task in the field of computer vision with wide-ranging applications in intelligent transportation systems, traffic monitoring, law enforcement, and access control. The goal is to accurately recognize and reconstruct vehicle license plates from images or video streams, often captured under challenging real-world conditions such as varying lighting, occlusions, motion blur, and diverse plate formats. Deep learning models, particularly convolutional neural networks, have significantly advanced the performance and reliability of car plate recognition. This project aims to explore and implement deep learning-based approaches for license plate recognition, emphasizing practical challenges and the impact of robust solutions in modern urban infrastructure and mobility management.

**Dataset:** [[2]](#2) [CCPD Dataset Repository](https://github.com/detectRecog/CCPD) | [Our Rebalanced Subsampled Dataset](https://drive.google.com/drive/folders/17b7I98G9W3TsrY_xIJaHYKCwosJNkgRX?usp=sharing)

The objective of this project is to design and implement a deep learning-based system for license plate recognition, following the methodology outlined in [[1]](#1). The proposed solution is structured as a two-stage pipeline, leveraging the strengths of different neural network architectures to address the distinct subtasks involved in the recognition process. In the first stage, a YOLOv5 model is employed for license plate detection, allowing for fast and accurate localization of the plate region within vehicle images, even under challenging environmental conditions. In the second stage, the cropped plate region is passed to a specialized recognition model based on the PDLPR architecture. This model is responsible for decoding the sequence of alphanumeric characters on the plate, effectively treating the task as a sequence prediction problem. The integration of these two components aims to deliver a robust and efficient system for plates recognition and reconstruction suitable for deployment in real-world scenarios.

The project was a collaborative effort with my colleague [Filippo Casini](https://github.com/Filippo-hub).

---
**Computer Vision Course, Sapienza University of Rome, Artificial Intelligence and Robotics Master Program**

---

## 🚀 Project Overview

This repository implements a comprehensive license plate recognition system using a two-stage deep learning pipeline:

1. **Detection Stage**: Fine-tuned YOLOv5 model for accurate license plate localization
2. **Recognition Stage**: PDLPR (Position-aware Deep License Plate Recognition) model for character sequence prediction

The complete pipeline achieves approximately **110 FPS** inference speed, making it suitable for real-time applications.

## 📁 Repository Structure

### 🔬 Main Implementation Files

#### `paper_implementation.ipynb` ⭐ **[MAIN IMPLEMENTATION]**
**The core implementation following the research paper architecture.**

This notebook contains the complete implementation of the PDLPR architecture as described in the research paper. Key components include:

- **YOLOv5 Inference Pipeline**: 
  - Loading and running the fine-tuned YOLOv5 model for license plate detection
  - Batch processing for train/validation/test sets
  - Automatic cropping of detected license plates

- **PDLPR Model Architecture**:
  - **Focus Module**: Efficient downsampling using patch concatenation technique
  - **Improved Global Feature Extractor (IGFE)**: CNN-based feature extraction with ResNet blocks
  - **Positional Encoding**: Spatial position awareness for better character localization
  - **Encoder Blocks**: Multi-head self-attention mechanism for sequence modeling
  - **CTC Head**: Connectionist Temporal Classification for variable-length sequence prediction

- **Training Infrastructure**:
  - Complete training loop with validation
  - Weights & Biases integration for experiment tracking
  - Learning rate scheduling and early stopping
  - Comprehensive evaluation metrics (accuracy, character-level accuracy, province/alphabet accuracy)

- **End-to-End Pipeline**:
  - `LicensePlateRecognitionPipeline` class for production-ready inference
  - Integration of YOLOv5 detection + PDLPR recognition
  - Real-time inference capabilities (~110 FPS)

- **Performance Analysis**:
  - Training curve visualization
  - Inference speed benchmarking
  - Sample prediction analysis

#### `detection_model.ipynb`
**baseline detection algorithm implementation.**

Contains the implementaion of the detection baseline algorithm

- Custom dataset
- Faster R-CNN model tuning for license plate detection
- Evaluation metrics and validation procedures
- Model export for inference pipeline

#### `baseline_detection_model+PDLPR.ipynb`
**Baseline comparison using Faster R-CNN + PDLPR.**

Implements an alternative detection approach for performance comparison:

- Faster R-CNN model implementation for license plate detection
- Integration with PDLPR recognition model
- Performance benchmarking against YOLOv5 approach
- Analysis of detection accuracy and speed trade-offs

#### `baseline_detection_model+recognition_model.ipynb`
**Classical CRNN baseline implementation.**

Traditional approach using Faster R-CNN for detection and CRNN for recognition:

- Faster R-CNN detection pipeline
- CRNN (Convolutional Recurrent Neural Network) architecture
- CTC loss implementation for sequence learning
- Baseline performance establishment for comparison with PDLPR

#### `baseline_ground_truth+recognition_model.ipynb`
**Recognition model evaluation using ground truth bounding boxes.**

Isolates recognition performance by using ground truth detections:

- Ground truth bounding box extraction from dataset annotations
- Pure recognition model evaluation
- Upper bound performance analysis
- Error analysis without detection noise

### 📊 Analysis and Utilities

#### `analysis/` Directory
- **`data_rebalancing.ipynb`**: Dataset analysis and class balancing strategies
- **`FineTuned_YOLOv5.ipynb`**: Detailed YOLOv5 fine-tuning experiments
- **`yolo_cropping.ipynb`**: Image preprocessing and cropping utilities
- **`yolo_finetune_create_dataset.py`**: Automated dataset creation for YOLO training

## 🛠️ Technical Implementation Details

### Model Architectures

#### PDLPR (Position-aware Deep License Plate Recognition)
- **Input**: 64×256 grayscale images
- **Architecture Components**:
  - Focus module for efficient spatial downsampling
  - ResNet-based feature extraction backbone
  - Positional encoding for spatial awareness
  - Multi-head attention encoder blocks
  - CTC decoder for sequence prediction
- **Training**: Adam optimizer, CTC loss, learning rate scheduling
- **Performance**: ~1ms inference per image

#### YOLOv5 Fine-tuning
- **Base Model**: YOLOv5s architecture
- **Modifications**: Single-class detection (license plates)
- **Training Data**: CCPD green plates subset
- **Performance**: ~10ms inference per image

### Dataset Processing
The CCPD (Chinese City Parking Dataset) contains over 300k images with encoded annotations. Our processing pipeline:

1. **Subset Selection**: Focus on green license plates for consistency due to harware limitations (CCPD_green)
2. **Data Balancing**: Address class imbalance in character distribution
3. **Format Conversion**: Transform to YOLO-compatible annotations
4. **Train/Val/Test Split**: 70/15/15 distribution

### Character Encoding
- **Chinese Provinces**: 34 different provincial characters
- **Letters**: 25 alphabetic characters (excluding I and O to avoid confusion)
- **Alphanumeric**: 35 characters (0-9, A-Z excluding I and O)
- **Total Vocabulary**: 69 unique characters + CTC blank token

## 📊 Results and Performance

### Pipeline Performance
- **Overall Accuracy**: Detailed in individual notebook evaluations
- **Inference Speed**: ~110 FPS (10ms YOLOv5 + 1ms PDLPR)
- **Character-level Accuracy**: Comprehensive evaluation in `paper_implementation.ipynb`

### Model Comparisons
1. **YOLOv5 + PDLPR**: Best overall performance and speed
2. **Faster R-CNN + PDLPR**: Slower inference
3. **Faster R-CNN + CRNN**: Baseline comparison model

## 🚦 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt #(file present in extra folder)
pip install ultralytics  # for YOLOv5
pip install wandb       # for experiment tracking
pip install matplotlib pillow opencv-python
```

### Quick Start
1. **Data Preparation**: Download the dataset from our [Google Drive link](https://drive.google.com/drive/folders/17b7I98G9W3TsrY_xIJaHYKCwosJNkgRX?usp=sharing)
2. **Model Training**: Run `paper_implementation.ipynb` for complete pipeline training
3. **Inference**: Use the `LicensePlateRecognitionPipeline` class for end-to-end inference

### Training Your Own Model
For detailed training procedures, refer to the individual notebook implementations, particularly `paper_implementation.ipynb` which contains the most comprehensive training pipeline.

## 📝 Experiments and Ablations

Each notebook represents different experimental configurations:
- **Detection Method Comparison**: YOLOv5 vs Faster R-CNN
- **Recognition Architecture**: PDLPR vs CRNN
- **Ground Truth Analysis**: Recognition performance without detection errors
- **Speed vs Accuracy Trade-offs**: Comprehensive benchmarking

---




## References
<a id="1">[1]</a> 
Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). 
A Real-Time License Plate Detection and
Recognition Model in Unconstrained Scenarios. Sensors, 24(9), 2791

<a id="2">[2]</a> 
Xu, Z.; Yang, W.; Meng, A.; Lu, N.; Huang, H.; Ying, C.; Huang, L.
Towards end-to-end license plate
detection and recognition: A large dataset and baseline. In Proceedings of the European Conference on
Computer Vision (ECCV), Munich, Germany, 8–14 September 2018.

<a id="3">[3]</a> 
 R. K. Prajapati, Y. Bhardwaj, R. K. Jain and D. Kamal Kant Hiran.
”A Review Paper on Automatic Number Plate Recognition using Machine Learning : An In-Depth Analysis of Machine Learning Techniques in
Automatic Number Plate Recognition: Opportunities and Limitations,”
2023 International Conference on
Computational Intelligence, Communication Technology and Networking (CICTN), Ghaziabad, India, 2023,
pp. 527-532




