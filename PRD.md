# Chest X-ray Pneumonia Demo 产品需求文档

## 1. 产品概述

Chest X-ray Pneumonia Demo 是一个基于Python和Streamlit构建的教育演示web应用，用于展示胸部X光片的肺炎检测功能。该应用仅限教育/研究用途，非临床诊断工具。

## 2. 核心功能

### 2.1 功能模块

| 模块名称 | 功能描述 |
|---------|---------|
| 图片上传 | 支持用户上传胸部X光图片（jpg/png格式） |
| 模型推理 | 使用Hugging Face预训练模型进行肺炎检测 |
| 结果展示 | 显示预测类别（Normal/Pneumonia）和置信度百分比 |
| 热图生成 | 如模型支持，生成Grad-CAM热图显示模型关注区域 |
| 免责声明 | 醒目显示应用仅用于教育演示的声明 |

### 2.2 核心流程

1. 用户访问应用
2. 浏览并上传胸部X光图片
3. 系统加载预训练模型
4. 模型对图片进行推理
5. 系统生成预测结果和热图（如果支持）
6. 显示结果给用户

## 3. 技术架构

### 3.1 技术栈

- 前端框架：Streamlit
- 后端语言：Python
- 深度学习框架：PyTorch
- 模型来源：Hugging Face
- 图像处理：OpenCV
- 可视化：Matplotlib
- 热图生成：pytorch-grad-cam（如果可用）

### 3.2 模型选择

1. 首选模型：ayushirathour/chest-xray-pneumonia-detection（MobileNetV2基底，模型小，CPU运行快）
2. 备用模型：dima806/chest_xray_pneumonia_detection（ViT基底）

### 3.3 系统架构

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
┌────────▼────────┐
│  模型加载与推理  │
└────────┬────────┘
         │
┌────────▼────────┐
│  热图生成（可选） │
└────────┬────────┘
         │
┌────────▼────────┐
│  结果展示与可视化 │
└─────────────────┘
```

## 4. UI设计

### 4.1 页面布局

1. 顶部：大标题 "Chest X-ray Pneumonia Demo"
2. 中部：
   - 醒目的红色粗体免责声明："本工具仅供教育演示，非医生诊断，请咨询专业医师"
   - 图片上传按钮
   - 上传图片预览
3. 底部：
   - 结果显示区（预测类别 + 置信度进度条）
   - 热图显示区（如果模型支持）

### 4.2 交互流程

1. 用户点击上传按钮选择图片
2. 系统显示上传的图片预览
3. 系统自动开始模型推理
4. 推理完成后显示结果和热图

## 5. 技术要求

1. 强制使用CPU模式：device = torch.device('cpu')
2. 全本地运行，图片不上传云端
3. 网络问题时配置代理：http://127.0.0.1:10808
4. 只做推理（inference），不训练
5. 输出结果格式：预测类别（Normal / Pneumonia） + 置信度百分比
6. 如模型支持且CPU能快速运行，生成Grad-CAM热图

## 6. 依赖项

- streamlit
- torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- transformers
- timm
- opencv-python
- matplotlib
- pytorch-grad-cam（如果可用）

## 7. 项目文件结构

```
├── app.py                # 主应用文件
├── requirements.txt      # 依赖文件
└── README.md             # 项目说明文档（中英双语）
```

## 8. 未来计划

- 后续可对模型进行finetune，提高检测准确率
- 添加更多疾病类型的检测
- 优化UI界面，提升用户体验
- 添加模型解释功能，增强教育价值

## 9. 伦理声明

本应用仅供教育和研究目的使用，不得用于临床诊断。所有检测结果仅供参考，不能替代专业医生的诊断。

## 10. 数据来源

模型训练数据来源于Kaggle Chest X-ray Pneumonia数据集。