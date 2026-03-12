## ================= 1. Model & Environment Setup =================
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

# Language dictionary for multi-language support
LANG = {
    'zh': {
        'title': '🩺 胸部X光片肺炎检测助手',
        'disclaimer': '⚠️ 免责声明',
        'disclaimer_text': '本工具仅用于教育和研究目的，不能替代专业医生的临床诊断。如有健康问题，请务必咨询专业医疗人员。',
        'loading_model': '正在加载模型...',
        'loaded_model': '成功加载模型: my_chest_xray_model.keras',
        'load_model_failed': '加载本地模型失败: {e}',
        'creating_placeholder': '正在创建占位模型...',
        'created_placeholder': '成功创建占位模型',
        'placeholder_note': '请注意：这是一个占位模型，预测结果仅供演示。请检查您的真实模型文件是否正确放置。',
        'model_load_error': '加载模型失败: {e}',
        'upload_title': '上传胸部X光片',
        'upload_placeholder': '请选择一张胸部X光片',
        'analyzing': '正在分析图像...',
        'detection_result': '检测结果',
        'diagnosis': '诊断结果: {diagnosis}',
        'confidence': '信心度: {confidence:.2f}%',
        'pneumonia': '肺炎',
        'normal': '正常',
        'ai_visualization': 'AI 分析可视化',
        'original_xray': '原始 X 光片',
        'grad_cam_heatmap': 'Grad-CAM 热力图',
        'heatmap_caption': '🔴 红色区域代表 AI 重点关注的病变特征位置',
        'no_conv_layer': '未找到卷积层，无法生成Grad-CAM热力图',
        'prediction_failed': '预测失败: {e}',
        'heatmap_failed': '生成热力图失败: {e}'
    },
    'en': {
        'title': '🩺 Chest X-ray Pneumonia Detection Assistant',
        'disclaimer': '⚠️ Disclaimer',
        'disclaimer_text': 'This tool is for educational and research purposes only and cannot replace clinical diagnosis by professional doctors. If you have health concerns, please consult professional medical personnel.',
        'loading_model': 'Loading model...',
        'loaded_model': 'Successfully loaded model: my_chest_xray_model.keras',
        'load_model_failed': 'Failed to load local model: {e}',
        'creating_placeholder': 'Creating placeholder model...',
        'created_placeholder': 'Successfully created placeholder model',
        'placeholder_note': 'Note: This is a placeholder model, prediction results are for demonstration only. Please check if your real model file is correctly placed.',
        'model_load_error': 'Failed to load model: {e}',
        'upload_title': 'Upload Chest X-ray',
        'upload_placeholder': 'Please select a chest X-ray',
        'analyzing': 'Analyzing image...',
        'detection_result': 'Detection Result',
        'diagnosis': 'Diagnosis: {diagnosis}',
        'confidence': 'Confidence: {confidence:.2f}%',
        'pneumonia': 'Pneumonia',
        'normal': 'Normal',
        'ai_visualization': 'AI Analysis Visualization',
        'original_xray': 'Original X-ray',
        'grad_cam_heatmap': 'Grad-CAM Heatmap',
        'heatmap_caption': '🔴 Red areas represent locations of pathological features focused on by AI',
        'no_conv_layer': 'No convolutional layer found, cannot generate Grad-CAM heatmap',
        'prediction_failed': 'Prediction failed: {e}',
        'heatmap_failed': 'Failed to generate heatmap: {e}'
    }
}

@st.cache_resource
def load_model(lang='en'):
    try:
        st.write(LANG[lang]['loading_model'])
        
        # Priority: Load locally fine-tuned VGG16 model
        try:
            model = tf.keras.models.load_model('my_chest_xray_model.keras', compile=False)
            
            # Manually compile model to ensure inference availability
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            st.write(LANG[lang]['loaded_model'])
            return model
        except Exception as e:
            st.warning(LANG[lang]['load_model_failed'].format(e=e))
            st.write(LANG[lang]['creating_placeholder'])
            
            # Create placeholder model to ensure application availability
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            st.write(LANG[lang]['created_placeholder'])
            st.info(LANG[lang]['placeholder_note'])
            return model
    except Exception as e:
        st.error(LANG[lang]['model_load_error'].format(e=e))
        return None

## ================= 2. Image Preprocessing Pipeline =================

def preprocess_image(image):
    # Convert to RGB mode to comply with VGG16's 3-channel input requirement
    image = image.convert('RGB')
    # Resize to VGG16's standard input size
    image = image.resize((224, 224))
    # Convert to numpy array
    img_array = np.array(image)
    # Normalize to [0,1] range, strictly aligning with VGG16's preprocessing standard on ImageNet
    img_array = img_array / 255.0
    # Add batch dimension to match model input format
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

## ================= 3. Inference Function =================

def predict(model, image, lang='en'):
    try:
        img_array = preprocess_image(image)
        # Execute model inference and get binary classification probability output
        prediction = model.predict(img_array)[0][0]
        return prediction
    except Exception as e:
        st.error(LANG[lang]['prediction_failed'].format(e=e))
        return None

## ================= 4. Grad-CAM Algorithm Implementation =================

def find_last_conv_layer(model):
    # Reverse traverse model layers to find the last convolutional layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def generate_grad_cam(model, img_array, layer_name=None, lang='en'):
    # Automatically locate target convolutional layer to ensure algorithm generality
    if layer_name is None:
        layer_name = find_last_conv_layer(model)
        if layer_name is None:
            st.warning(LANG[lang]['no_conv_layer'])
            return None
    
    # Build model mapping feature maps to prediction results
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Use gradient tape to calculate gradients of class with respect to feature maps
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0][0]  # For binary classification, focus on sigmoid output
    
    # Calculate gradients and perform global average pooling to get feature importance weights
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate heatmap: weighted sum of feature maps and weights
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # ReLU activation to filter out negative contributions, enhancing visualization effect
    heatmap = tf.maximum(heatmap, 0)
    
    # Normalize to ensure consistent heatmap range
    if tf.reduce_max(heatmap) > 0:
        heatmap /= tf.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4, lang='en'):
    try:
        # Resize heatmap to match original image dimensions
        heatmap = cv2.resize(heatmap, (image.width, image.height))
        
        # Convert heatmap to color Jet mapping
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy array
        img_array = np.array(image)
        
        # Ensure dimension matching
        if img_array.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Ensure channel count matching
        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Overlay heatmap with alpha=0.4 to ensure underlying anatomical structures remain visible
        overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
        
        return Image.fromarray(overlay)
    except Exception as e:
        st.error(LANG[lang]['heatmap_failed'].format(e=e))
        return image

## ================= 5. Streamlit UI & Main Execution =================

def main():
    # Language selection
    lang = st.sidebar.selectbox(
        "Language / 语言",
        options=['en', 'zh'],
        format_func=lambda x: 'English' if x == 'en' else '中文'
    )
    
    st.set_page_config(
        page_title=LANG[lang]['title'],
        page_icon="🩺",
        layout="wide"
    )
    
    st.title(LANG[lang]['title'])
    
    # Red disclaimer
    st.markdown(f"""
    <div style="background-color: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336;">
        <h4 style="color: #c62828;">{LANG[lang]['disclaimer']}</h4>
        <p style="color: #b71c1c;">{LANG[lang]['disclaimer_text']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model(lang)
    
    if model is not None:
        st.subheader(LANG[lang]['upload_title'])
        uploaded_file = st.file_uploader(LANG[lang]['upload_placeholder'], type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            with st.spinner(LANG[lang]['analyzing']):
                prediction = predict(model, image, lang)
                
                if prediction is not None:
                    st.subheader(LANG[lang]['detection_result'])
                    
                    # Use 0.5 as binary classification cutoff (may optimize based on ROC curve in the future)
                    if prediction > 0.5:
                        diagnosis = LANG[lang]['pneumonia']
                        confidence = prediction * 100
                        color = "#f44336"
                    else:
                        diagnosis = LANG[lang]['normal']
                        confidence = (1 - prediction) * 100
                        color = "#4caf50"
                    
                    st.markdown(f"""
                    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 10px 0;">
                        <h3 style="color: {color};">{LANG[lang]['diagnosis'].format(diagnosis=diagnosis)}</h3>
                        <div style="width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                            <div style="width: {confidence}%; height: 100%; background-color: {color}; border-radius: 10px;"></div>
                        </div>
                        <p style="margin-top: 10px; color: #333333;">{LANG[lang]['confidence'].format(confidence=confidence)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate and display Grad-CAM heatmap
                    img_array = preprocess_image(image)
                    heatmap = generate_grad_cam(model, img_array, lang=lang)
                    
                    if heatmap is not None:
                        overlay_image = overlay_heatmap(image, heatmap, lang=lang)
                        
                        st.subheader(LANG[lang]['ai_visualization'])
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(image, caption=LANG[lang]['original_xray'], width=400)
                        
                        with col2:
                            st.image(overlay_image, caption=LANG[lang]['grad_cam_heatmap'], width=400)
                            st.caption(LANG[lang]['heatmap_caption'])

if __name__ == "__main__":
    main()