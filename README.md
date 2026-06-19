# TinyML Sine Wave Predictor
## Overview
A lightweight, from-scratch machine learning pipeline demonstrating the core principles of TinyML and edge AI. This project bridges the gap between high-level model training and low-level hardware execution by training a neural network in Python and deploying it as a pure C++ inference engine. It is designed to run on severely resource-constrained microcontrollers (such as an Arduino Nano or ESP32) without relying on any external machine learning frameworks.

## Key Features & TinyML Highlights

### Zero-Dependency Inference:
The deployment code runs entirely on standard C++ math operations. By eliminating heavy libraries like TensorFlow Lite or PyTorch, the executable size is kept to an absolute minimum.

### Ultra-Low Memory Footprint:
Proves the ability to operate within strict hardware constraints (e.g., <256KB of RAM). Model weights and biases are statically allocated in memory at compile time, eliminating dynamic memory allocation overhead.

### Transparent Architecture:
Demystifies the "black box" of deep learning by manually implementing the forward pass — translating high-level API calls into explicit matrix multiplications, dot products, and ReLU activations.

## The Pipeline

### Model Training (Python & TensorFlow):
A compact Multi-Layer Perceptron (MLP) is constructed and trained on synthetic data to predict continuous sine wave values.

### Parameter Extraction:
A custom Python script extracts the optimized weights and biases from the trained model and automatically formats them into a strictly typed C++ header file (weights.h).

### Edge Execution (C++):
A custom-built inference engine (main.cpp) ingests the static weight matrices and computes predictions locally, replicating the exact math of the TensorFlow backend.

## Why This Matters ?
This project serves as a practical proof-of-concept for <b>Edge AI</b>. It demonstrates a deep, fundamental understanding of neural network mathematics and the practical engineering skills required to deploy intelligent systems directly onto physical, low-power edge devices.
