# Chest X-ray Pneumonia Assistant 🫁

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live_Demo-blue)](https://huggingface.co/spaces/krischan-ai/Chest-Xray-Pneumonia-Assistant)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model_Training_Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/dingzhendinner/chest-x-ray-images)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)

## 中文版本

### 📖 项目描述

**Chest X-ray Pneumonia Assistant** 是一个基于 Python 和 Streamlit 构建的医疗影像辅助 Web 演示应用。该项目通过部署**本地微调 (Fine-tuned) 的深度学习模型**，展示了 AI 在胸部 X 光片肺炎特征检测中的应用潜力。
*(注：本应用仅供教育、学术交流与研究演示使用，非临床诊断工具。)*

![App UI Screenshot](assets/app_screenshot.png)

### 🛠️ 技术栈

- **前端框架**：Streamlit
- **后端语言**：Python
- **深度学习框架**：TensorFlow / Keras
- **核心大脑**：本地微调的 VGG16 模型 (Fine-tuned VGG16)
- **图像处理**：PIL / NumPy
- **运行环境**：纯 CPU 推理优化 (`tensorflow-cpu`)

### 🚀 版本演进与技术重构

#### ✨ [v2.1 最新升级] - 可解释性 AI (XAI) 与病灶热力图可视化
* **Grad-CAM 算法集成**：彻底打破深度学习的“黑盒”限制。系统通过计算 VGG16 最终卷积层（如 `block5_conv3`）的梯度，实时动态生成类激活热力图 (Heatmap)。
* **临床信任度提升**：以红色高亮直观标注 AI 做出“肺炎”预测时所关注的肺部浑浊区域。这不仅能辅助放射科医生进行直观的交叉验证，更能有效验证模型是否真正学到了病理特征，而非虚假相关性（Spurious Correlations）。
* **新增技术栈**：OpenCV (图像通道处理与彩色映射), Matplotlib。

#### [v2.0] - 本地模型 Fine-tuning 与部署闭环
在项目的初始阶段（v1.0），本 Demo 采用了 Hugging Face 上的轻量级预训练模型进行快速原型开发。为了进一步掌控模型的训练细节并优化在特定医疗数据集上的表现，v2.0 版本进行了核心引擎的重构：

* **核心大脑替换**：模型后端从 PyTorch 切换为 `TensorFlow/Keras`，并部署了我在 Kaggle 环境下基于 **Chest X-Ray Images (Pneumonia)** 数据集亲手训练的 VGG16 微调（Fine-tuned）模型。

![VGG16 Fine-tuning Architecture](assets/model_architecture.png)

* **推理性能优化**：为了保证在普通个人电脑上的流畅运行体验，环境配置中指定了 `tensorflow-cpu`，在不依赖 GPU 的情况下依然能实现毫秒级的快速推断。

> 💡 **开发者笔记 / 技术复盘**
>
> 在这次从数据预处理、模型训练到最终 Web 部署的完整闭环中，我深刻体会到了**迁移学习 (Transfer Learning)** 在医疗影像等相对小样本数据集上的巨大价值。
>
> 在训练策略上，为了防止模型在有限的 X 光图片上产生过拟合 (Overfitting)，我刻意**冻结了 VGG16 骨干网络中高达 1400 多万个底层参数** (Non-trainable params)，仅开放并集中算力训练我自定义的分类顶层 (约 6.5 万个 Trainable params)。

![Training Accuracy and Loss Curves](assets/training_curves.png)
>
> 实践证明，这一"以静制动"的策略不仅成倍缩短了单次 Epoch 的训练时间，更重要的是，它完美保留了模型在 ImageNet 上学到的泛化视觉特征提取能力。配合 Early Stopping 等回调机制，模型在未见过的测试集上展现出了极高的召回率 (Recall)。

![Confusion Matrix on Test Set](assets/confusion_matrix.png)

这次重构不仅提升了 Demo 的准确度，也让我对 AI 算法在医疗场景下的工程落地有了更直观、更底层的认知。

> 关于 v2.1 的思考：在纯粹的端到端评估后，我意识到医疗 AI 真正落地的最大阻碍是"信任鸿沟"。引入 Grad-CAM 是我主动向临床实用性迈出的一步。看到热力图精准覆盖在测试集图片的肺部浑浊处，而非图片边缘的字母标记上，这比单纯达到 90% 的 AUC 更让我确信模型的鲁棒性。

![ROC Curve and AUC](assets/roc_curve.png)

### ✨ 功能特点

1. **图片上传**：支持上传胸部 X 光图片 (JPG/PNG)，内置防报错机制（灰度图自动转换为 RGB 格式）。
2. **极速推理**：使用本地 Fine-tuned Keras 模型进行特征提取与分类。
3. **结果展示**：直观显示预测类别 (Normal / Pneumonia) 及置信度百分比。
4. **隐私保护**：全本地运行，图片数据不上云，严格保护数据隐私。

### 💻 安装与运行

#### 🌟 在线体验 (Live Demo)
您无需在本地配置任何环境，直接点击下方链接即可在浏览器中体验完整功能：
👉 **https://huggingface.co/spaces/krischan-ai/Chest-Xray-Pneumonia-Assistant**

*(如果您希望在本地运行，请参考以下步骤：)*

#### 1. 安装依赖
建议在虚拟环境中运行。
```bash
pip install -r requirements.txt
```

#### 2. 运行应用
```bash
streamlit run app.py
```

### 🔮 未来计划

- [ ] **消除数据偏差 (Mitigating Data Bias)**：整合国内本土或亚洲人群的医疗影像标注数据，进行下一阶段的模型 Fine-tuning，以降低当前开源数据集（主要源于欧美医疗系统）带来的地域与种族特征偏差。
- [x] **可解释性 AI (Explainable AI / XAI)**：[已于 v2.1 版本完成实现] 探索集成 LIME 或手写基于 Keras 的 Grad-CAM 算法，恢复并优化模型的可解释性（热力图）功能，辅助医生理解 AI 的空间特征决策逻辑。
- [ ] **多分类扩展 (Multi-class Extension)**：扩展多分类任务，将检测范围从单一的肺炎扩充至包含肺结核、气胸等多种肺部病变的综合筛查。

### ⚖️ 伦理与免责声明

[🚨 仅供教育用途]
本应用的所有预测结果仅由机器学习算法生成，不得用于临床诊断。模型输出不能、也不应替代专业放射科医师或医疗专业人士的医学判断。如有健康问题，请务必咨询合规的医疗机构。

**数据来源：**
初始模型训练数据来源于公开的 Kaggle Chest X-ray Pneumonia 数据集。

---

## English Version

### 📖 Project Description

**Chest X-ray Pneumonia Assistant** is an educational web application built with Python and Streamlit. By deploying a **custom fine-tuned deep learning model**, this project demonstrates the potential of AI in detecting pneumonia features from chest X-ray images.
*(Note: This application is strictly for educational, academic, and research demonstration purposes. It is not a clinical diagnostic tool.)*

![App UI Screenshot](assets/app_screenshot.png)

### 🛠️ Technical Implementation

- **Frontend**：Streamlit
- **Backend**：Python
- **DL Framework**：TensorFlow / Keras
- **Model**：Locally fine-tuned VGG16
- **Image Processing**：PIL / NumPy
- **Environment**：CPU-optimized inference (`tensorflow-cpu`)

### 🚀 Version Evolution & Refactoring

#### ✨ [v2.1 Latest Update] - Explainable AI (XAI) & Lesion Heatmap Visualization
* **Grad-CAM Integration**: Shattered the "black-box" limitation of standard CNNs. By computing the gradients of the final convolutional layer (e.g., `block5_conv3`), the system dynamically generates Class Activation Heatmaps in real-time.
* **Enhancing Clinical Trust**: Visually highlights the exact lung opacities driving the AI's "Pneumonia" prediction with red overlays. This provides radiologists with an intuitive second opinion for cross-validation and ensures the model is learning true pathological features rather than spurious correlations.
* **New Tech Stack**: OpenCV (Color mapping and channel processing), Matplotlib.

#### [v2.0] - Local Model Fine-tuning & Deployment Loop
In the initial phase (v1.0), this demo utilized a lightweight pre-trained model from Hugging Face for rapid prototyping. To gain deeper control over training details and optimize performance on specific medical datasets, version 2.0 refactored the core engine:

* **Core Engine Replacement**: Switched the backend from PyTorch to `TensorFlow/Keras`, deploying a custom Fine-tuned VGG16 model that I personally trained on Kaggle using the **Chest X-Ray Images (Pneumonia)** dataset.

![VGG16 Fine-tuning Architecture](assets/model_architecture.png)

* **Inference Optimization**: To ensure a smooth user experience on standard personal computers, `tensorflow-cpu` was specified in the environment configuration, achieving millisecond-level rapid inference without relying on a GPU.

> 💡 **Developer's Reflection**
>
> Through this complete end-to-end cycle—from data preprocessing and model training to final web deployment—I gained profound insights into the immense value of **Transfer Learning** on relatively small medical imaging datasets.
>
> Regarding the training strategy, to prevent Overfitting on a limited number of X-ray images, I deliberately **froze over 14 million base parameters** (Non-trainable params) in the VGG16 backbone, concentrating the computational power exclusively on training my custom top classification layer (approx. 65,000 Trainable params).

![Training Accuracy and Loss Curves](assets/training_curves.png)
>
> Practice proved that this strategy not only drastically reduced the training time per Epoch but, more importantly, perfectly retained the generalized visual feature extraction capabilities the model learned from ImageNet. Coupled with mechanisms like Early Stopping, the model achieved an exceptionally high Recall rate on the unseen test set.

![Confusion Matrix on Test Set](assets/confusion_matrix.png)

This refactoring not only enhanced the demo's accuracy but also deepened my foundational understanding of engineering and deploying AI algorithms in practical medical scenarios.

Reflection on v2.1: After pure end-to-end evaluation, I realized the biggest hurdle for medical AI is the "trust gap". Integrating Grad-CAM was a proactive step toward clinical viability. Seeing the heatmaps accurately highlight lung opacities rather than peripheral image markers gave me more confidence in the model's robustness than merely achieving a 90% AUC.

![ROC Curve and AUC](assets/roc_curve.png)

### ✨ Features

1. **Image Upload**: Supports uploading chest X-ray images (JPG/PNG), with built-in error prevention (automatic conversion of grayscale images to RGB format).
2. **Fast Inference**: Uses a local fine-tuned Keras model for feature extraction and classification.
3. **Result Display**: Intuitively shows predicted class (Normal / Pneumonia) and confidence percentage.
4. **Privacy Protection**: Runs entirely locally, no image data is uploaded to the cloud, strictly protecting data privacy.

### 💻 Installation & Usage

#### 🌟 Live Demo
You can experience the full functionality directly in your browser without any local environment setup:
👉 **https://huggingface.co/spaces/krischan-ai/Chest-Xray-Pneumonia-Assistant**

*(If you prefer to run it locally, please follow the steps below:)*

#### 1. Install Dependencies
It is recommended to run in a virtual environment.
```bash
pip install -r requirements.txt
```

#### 2. Launch the App
```bash
streamlit run app.py
```

### 🔮 Future Plans

- [ ] **Mitigating Data Bias**: Integrate local or broader Asian medical imaging datasets for the next phase of fine-tuning, aiming to reduce regional and demographic biases inherent in the current open-source dataset (which is primarily Western-centric).
- [x] **Explainable AI (XAI)**: [Successfully implemented in v2.1] Explore the integration of LIME or custom Keras-based Grad-CAM algorithms to restore and optimize the model's interpretability (heatmaps), assisting medical professionals in understanding the AI's spatial decision-making logic.
- [ ] **Multi-class Extension**: Expand the binary classification task to a multi-class framework, broadening the detection scope to include comprehensive screening for various lung lesions such as tuberculosis and pneumothorax.

### ⚖️ Ethical Statement & Disclaimer

[🚨 Educational Use Only]
All prediction results from this application are generated solely by machine learning algorithms and should not be used for clinical diagnosis. Model outputs cannot and should not replace the medical judgment of professional radiologists or healthcare professionals. If you have health concerns, please consult a qualified medical institution.

**Data Source:**
The initial model training data is sourced from the public Kaggle Chest X-ray Pneumonia dataset.
