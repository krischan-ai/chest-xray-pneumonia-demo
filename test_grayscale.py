import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import numpy as np
import os

# 代理配置已移除

# 强制使用CPU
device = torch.device('cpu')

# 模型名称
MODEL_NAME = "dima806/chest_xray_pneumonia_detection"

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
        print(f"加载模型失败: {e}")
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

# 主函数
if __name__ == "__main__":
    # 加载模型
    model, image_processor = load_model()
    
    if model is not None:
        print("模型加载成功！")
        
        # 创建一个灰度测试图像
        print("创建灰度测试图像...")
        # 创建一个224x224的灰度图像（单通道）
        gray_image = Image.new('L', (224, 224), color=128)  # 'L'模式表示灰度图
        print(f"创建的图像模式: {gray_image.mode}, 大小: {gray_image.size}")
        
        # 测试预测
        print("测试预测...")
        try:
            predicted_class, confidence = predict(gray_image, model, image_processor)
            print(f"预测结果: {CLASS_LABELS[predicted_class]}, 置信度: {confidence:.2f}%")
            print("测试成功！灰度图已成功转换为RGB并处理。")
        except Exception as e:
            print(f"预测失败: {e}")
    else:
        print("模型加载失败，无法测试。")