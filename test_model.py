import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os

# 代理配置已移除

# 强制使用CPU
device = torch.device('cpu')

# 模型名称
MODEL_NAME = "ayushirathour/chest-xray-pneumonia-detection"
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
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs

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
        print("测试完成。")
    else:
        print("模型加载失败。")