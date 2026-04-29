# Java_Project_Neuro
简单的卷积神经网络
# Chinese Zodiac Image Classifier - CNN from Scratch (Java)

A convolutional neural network implemented in pure Java with no deep learning libraries, designed to classify images of the 12 Chinese Zodiac animals.

## Requirements
- Java 8+
- Python 3.x + Pillow (for image preprocessing only)

## Getting Started

**Step 1: Generate dataset**
```bash
python resize.py
```
Resizes the 12 original images in `raw/` to 64×64 and saves them to `images/`.

**Step 2: Run**

Open the project in IntelliJ or VSCode and run the `main` method in `train.java`.

## Network Architecture
