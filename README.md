# Satellite Image Classification (EuroSAT)

## Project Overview
This project applies deep learning to classify satellite images into different land use and land cover categories using the EuroSAT dataset.  
The goal is to build a model capable of recognizing scenes such as forests, residential areas, highways, rivers, and agricultural land.  

A Streamlit web application is included, allowing users to upload satellite images and receive real-time predictions.

---

## Dataset
- **Name:** [EuroSAT Dataset](https://github.com/phelber/eurosat)  
- **Size:** 27,000 labeled images  
- **Classes (10 total):**  
  - Annual Crop  
  - Forest  
  - Herbaceous Vegetation  
  - Highway  
  - Industrial  
  - Pasture  
  - Permanent Crop  
  - Residential  
  - River  
  - Sea/Lake  

---

## Project Workflow
1. **Exploratory Data Analysis (EDA):**  
   - Data quality checks (completeness, uniqueness, consistency, etc.)  
   - Class distribution visualization  
   - Sample image visualization  

2. **Data Preprocessing:**  
   - Image resizing and normalization  
   - Train-test-validation split  
   - Data augmentation to reduce overfitting  

3. **Model Development:**  
   - Convolutional Neural Networks (CNNs) with TensorFlow/Keras  
   - Training with categorical cross-entropy loss  
   - Evaluation with accuracy, confusion matrix, and classification report  

4. **Deployment (Streamlit App):**  
   - Upload satellite images through UI  
   - Model predicts class label  
   - Displays confidence scores and example predictions  

---

## Technologies Used
- Python 3.10+  
- TensorFlow / Keras – Deep learning framework  
- NumPy, Pandas – Data handling  
- Matplotlib, Seaborn – Visualization  
- Streamlit – Web app for deployment  

---
### Author

Developed by Reyam Saleh Albalihi
Senior Computer Science Student | AI and Machine Learning Enthusiast | Data Scientist
