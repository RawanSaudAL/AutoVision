
# AutoVision: Real-Time Traffic Sign, Vehicle, and Pedestrian Detection

AutoVision is a high-performance object detection system that leverages YOLOv8 to accurately identify traffic signs, vehicles, and pedestrians. Designed for autonomous driving and urban traffic management, AutoVision aims to enhance safety and efficiency through robust real-time detection across diverse environmental conditions.

# Project Overview

- Objective: Enable real-time detection and classification of traffic signs, vehicles, and pedestrians to support autonomous driving and improve traffic safety.
- Dataset: Sourced from Roboflow, containing 7,000+ annotated images for traffic signs, cars, and pedestrians, with diverse lighting and weather conditions.

# Key Features

1. Data Preprocessing:
   - Resize, normalize, and apply data augmentation to images.
   - Encode bounding boxes and batch data for optimized loading during training.

2. YOLOv8 Model:
   - Utilizes YOLOv8, fine-tuned for traffic object detection with 25 epochs and an 800-pixel image size to balance resolution and speed.
   - Trained on Google Colab with a T4 GPU for efficient processing.

3. Performance Evaluation:
   - Evaluates model using Precision, Recall, F1-score, and IoU to assess accuracy and effectiveness in detecting traffic signs, vehicles, and pedestrians.

# Results

- Precision-Recall Curve: Demonstrates a mean Average Precision (mAP) of 0.824, showing strong model accuracy.
- High Recall Rate: Achieves a recall of 0.91, indicating reliable object detection under varied conditions.
- Real-World Scenarios: Tested successfully across diverse lighting, weather, and urban settings, with precision up to 98%.


# Usage

1. Open `train_yolov8_trafic_sign_detection.ipynb` in Jupyter Notebook.
2. Follow the steps to load data, train the model, and test its performance on the dataset.

# Future Enhancements

- Expanded Detection Classes: Add detection for additional road users like bicycles and motorcycles.
- Edge Optimization: Further optimize for deployment on mobile and edge devices.
- Adaptation to Specific Environments: Fine-tune for specific urban or highway settings to improve robustness.
