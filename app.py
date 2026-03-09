import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoModelForImageClassification, AutoImageProcessor
import os
import sys

# 代理配置已移除

# 强制使用CPU
device = torch.device('cpu')

# 模型名称（首选模型不可用，直接使用备用模型）
MODEL_NAME = "dima806/chest_xray_pneumonia_detection"
BACKUP_MODEL_NAME = "dima806/chest_xray_pneumonia_detection"

# 类别标签
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}

# 尝试加载模型
def load_model():
    try:
        print("正在加载模型...")
        model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        print(f"成功加载模型: {MODEL_NAME}")
        return model, image_processor
    except Exception as e:
        print(f"加载首选模型失败: {e}")
        print("尝试加载备用模型...")
        try:
            model = AutoModelForImageClassification.from_pretrained(BACKUP_MODEL_NAME)
            image_processor = AutoImageProcessor.from_pretrained(BACKUP_MODEL_NAME)
            print(f"成功加载备用模型: {BACKUP_MODEL_NAME}")
            return model, image_processor
        except Exception as e:
            print(f"加载备用模型失败: {e}")
            return None, None

# 图像预处理
def preprocess_image(image, image_processor):
    try:
        # 确保image是PIL Image对象
        if not isinstance(image, Image.Image):
            # 如果是numpy array，先转成PIL Image
            image = Image.fromarray(image)
        
        # 将灰度图转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 打印调试信息
        print(f"Image mode: {image.mode}, size: {image.size}")
        
        # 处理图像
        inputs = image_processor(images=image, return_tensors="pt")
        return inputs
    except Exception as e:
        print(f"预处理图像时出错: {e}")
        print(f"Image type: {type(image)}")
        if isinstance(image, Image.Image):
            print(f"Image mode: {image.mode}, size: {image.size}")
        raise

# 模型推理
def predict(image, model, image_processor):
    inputs = preprocess_image(image, image_processor)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item() * 100
    return predicted_class, confidence

# 由于pytorch-grad-cam安装失败，使用默认的文字说明
grad_cam_available = False

# 生成Grad-CAM热图（当前不可用）
def generate_cam(image, model, image_processor, predicted_class):
    return None

# 命令行界面
def run_command_line():
    print("Chest X-ray Pneumonia Demo")
    print("=" * 50)
    print("本工具仅供教育演示，非医生诊断，请咨询专业医师")
    print("=" * 50)
    
    # 加载模型
    model, image_processor = load_model()
    
    if model is None:
        print("模型加载失败，无法继续")
        return
    
    # 询问用户输入图片路径
    image_path = input("请输入胸部X光图片路径: ")
    
    try:
        # 打开图片
        image = Image.open(image_path)
        print("图片加载成功")
        
        # 预测
        print("正在分析...")
        predicted_class, confidence = predict(image, model, image_processor)
        
        # 显示结果
        print("\n预测结果:")
        print(f"类别: {CLASS_LABELS[predicted_class]}")
        print(f"置信度: {confidence:.2f}%")
        print("模型关注肺部区域")
        
    except Exception as e:
        print(f"处理图片失败: {e}")

# Streamlit应用
if __name__ == "__main__":
    # 如果是命令行运行
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_command_line()
    else:
        # 尝试运行Streamlit应用
        try:
            import streamlit as st
            
            st.title("Chest X-ray Pneumonia Demo")
            
            # 免责声明
            st.markdown("<h3 style='color: red; font-weight: bold;'>本工具仅供教育演示，非医生诊断，请咨询专业医师</h3>", unsafe_allow_html=True)
            
            # 加载模型
            model, image_processor = load_model()
            
            # 上传图片
            uploaded_file = st.file_uploader("选择胸部X光图片", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None and model is not None:
                # 显示上传的图片
                image = Image.open(uploaded_file)
                st.image(image, caption="上传的胸部X光图片", use_container_width=True)
                
                # 预测
                st.write("正在分析...")
                predicted_class, confidence = predict(image, model, image_processor)
                
                # 显示结果
                st.subheader("预测结果")
                st.write(f"**类别**: {CLASS_LABELS[predicted_class]}")
                st.write(f"**置信度**: {confidence:.2f}%")
                
                # 显示进度条
                st.progress(confidence / 100)
                
                # 生成并显示热图
                st.subheader("模型关注区域")
                cam_image = generate_cam(image, model, image_processor, predicted_class)
                if cam_image is not None:
                    st.image(cam_image, caption="Grad-CAM热图", use_container_width=True)
                else:
                    st.write("模型关注肺部区域")
            
            # 页脚
            st.markdown("---")
            st.write("© 2026 Chest X-ray Pneumonia Demo | 仅限教育/研究用途")
        except ImportError:
            print("Streamlit未安装，使用命令行模式")
            run_command_line()