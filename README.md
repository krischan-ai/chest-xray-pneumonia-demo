# Chest X-ray Pneumonia Demo

## 项目描述

Chest X-ray Pneumonia Demo 是一个基于Python和Streamlit构建的教育演示web应用，用于展示胸部X光片的肺炎检测功能。该应用使用Hugging Face上的预训练模型进行推理，仅用于教育和研究目的，非临床诊断工具。

## 安装运行步骤

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行应用

```bash
python -m streamlit run app.py
```

## 功能特点

1. **图片上传**：支持用户上传胸部X光图片（jpg/png格式），支持灰度图自动转换为RGB
2. **模型推理**：使用预训练模型进行肺炎检测
3. **结果展示**：显示预测类别（Normal/Pneumonia）和置信度百分比
4. **免责声明**：醒目显示应用仅用于教育演示的声明
5. **全本地运行**：图片不上传云端，保护隐私

## 技术实现

- **前端框架**：Streamlit
- **后端语言**：Python
- **深度学习框架**：PyTorch
- **模型来源**：Hugging Face（使用 dima806/chest_xray_pneumonia_detection 模型）
- **图像处理**：OpenCV
- **可视化**：Matplotlib

## 伦理声明

本应用仅供教育和研究目的使用，不得用于临床诊断。所有检测结果仅供参考，不能替代专业医生的诊断。

## 数据来源

模型训练数据来源于Kaggle Chest X-ray Pneumonia数据集。

## 未来计划

- 后续可对模型进行finetune，提高检测准确率
- 添加更多疾病类型的检测
- 优化UI界面，提升用户体验
- 添加模型解释功能，增强教育价值

---

# Chest X-ray Pneumonia Demo

## Project Description

Chest X-ray Pneumonia Demo is an educational web application built with Python and Streamlit, designed to demonstrate pneumonia detection from chest X-ray images. The application uses pre-trained models from Hugging Face for inference, and is intended for educational and research purposes only, not for clinical diagnosis.

## Installation and Running Steps

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python -m streamlit run app.py
```

## Features

1. **Image Upload**：Supports uploading chest X-ray images (jpg/png format), with automatic conversion of grayscale images to RGB
2. **Model Inference**：Uses pre-trained models for pneumonia detection
3. **Result Display**：Shows predicted class (Normal/Pneumonia) and confidence percentage
4. **Disclaimer**：Prominently displays that the application is for educational demonstration only
5. **Local Execution**：Images are not uploaded to the cloud, protecting privacy

## Technical Implementation

- **Frontend Framework**：Streamlit
- **Backend Language**：Python
- **Deep Learning Framework**：PyTorch
- **Model Source**：Hugging Face (using dima806/chest_xray_pneumonia_detection model)
- **Image Processing**：OpenCV
- **Visualization**：Matplotlib

## Ethical Statement

This application is for educational and research purposes only and should not be used for clinical diagnosis. All detection results are for reference only and cannot replace professional medical diagnosis.

## Data Source

The model training data is sourced from the Kaggle Chest X-ray Pneumonia dataset.

## Future Plans

- Fine-tune the model to improve detection accuracy
- Add detection for more disease types
- Optimize the UI interface to enhance user experience
- Add model explanation functionality to increase educational value