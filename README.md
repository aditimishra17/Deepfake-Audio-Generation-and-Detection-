# Deepfake-Audio-Generation-and-Detection-
Developed a deepfake audio pipeline using state-of-the-art neural architectures to both synthesize and detect synthetic speech. Utilized Tacotron 2 and WaveGlow for generating high-quality audio from text, and built a CNN-based classifier for detecting fake audio with 85% accuracy. Employed advanced audio features like MFCCs and spectrograms, and implemented data preprocessing to reduce false positives. The project demonstrates a complete cycle of deepfake audio creation and defense — applicable in voice authentication, media forensics, and AI ethics research.

This project focuses on detecting deepfake (synthetic) speech using a CNN-based classifier.
The model distinguishes between genuine and synthetic audio using advanced acoustic features like MFCCs and spectrograms.

Dataset: Fake audio samples were sourced from the ASVspoof 2019 dataset, which contains both genuine and spoofed speech generated using multiple synthesis and voice conversion techniques.
The goal is to build a robust detection pipeline that reduces false positives and generalizes well to unseen attacks.

Features
CNN-based binary classifier for deepfake detection.  
Feature extraction: MFCCs and log-mel spectrograms.
Data preprocessing: silence trimming, normalization, feature scaling.
85% classification accuracy on test data.
Evaluation metrics: Accuracy, Precision, Recall, F1-score.

Tech Stack
Python 3.10+
PyTorch – deep learning framework.
Librosa – audio processing and feature extraction.
NumPy / Pandas – data handling.
Matplotlib / Seaborn – visualizations.
Scikit-learn – preprocessing and evaluation.


Author: Aditi Mishra
linkedin: https://www.linkedin.com/in/aditi-mishra-312666259/
Gmail: amambm504@gmail.com
