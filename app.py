import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 加载本地模型
@st.cache_resource
def load_model():
    try:
        st.write("正在加载模型...")
        
        # 尝试加载本地模型
        try:
            model = tf.keras.models.load_model('my_chest_xray_model.keras', compile=False)
            
            # 手动编译模型
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            st.write("成功加载模型: my_chest_xray_model.keras")
            return model
        except Exception as e:
            st.warning(f"加载本地模型失败: {e}")
            st.write("正在创建占位模型...")
            
            # 创建一个简单的占位模型
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # 编译模型
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            st.write("成功创建占位模型")
            st.info("请注意：这是一个占位模型，预测结果仅供演示。请检查您的真实模型文件是否正确放置。")
            return model
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None

# 预处理图像
def preprocess_image(image):
    # 转换为RGB模式
    image = image.convert('RGB')
    # 调整尺寸为224x224
    image = image.resize((224, 224))
    # 转换为numpy数组
    img_array = np.array(image)
    # 归一化
    img_array = img_array / 255.0
    # 增加batch维度
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 预测函数
def predict(model, image):
    try:
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)[0][0]
        return prediction
    except Exception as e:
        st.error(f"预测失败: {e}")
        return None

# 主函数
def main():
    st.set_page_config(
        page_title="胸部X光片肺炎检测助手",
        page_icon="🩺",
        layout="wide"
    )
    
    # 标题
    st.title("🩺 胸部X光片肺炎检测助手")
    
    # 红色免责声明
    st.markdown("""
    <div style="background-color: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336;">
        <h4 style="color: #c62828;">⚠️ 免责声明</h4>
        <p style="color: #b71c1c;">本工具仅用于教育和研究目的，不能替代专业医生的临床诊断。如有健康问题，请务必咨询专业医疗人员。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 加载模型
    model = load_model()
    
    if model is not None:
        # 上传区域
        st.subheader("上传胸部X光片")
        uploaded_file = st.file_uploader("请选择一张胸部X光片", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # 创建左右两列布局
            col1, col2 = st.columns(2)
            
            with col1:
                # 显示上传的图像
                image = Image.open(uploaded_file)
                st.image(image, caption="上传的X光片", width=600)
            
            with col2:
                # 自动检测，不需要点击按钮
                with st.spinner("正在分析图像..."):
                    prediction = predict(model, image)
                    
                    if prediction is not None:
                        # 显示结果
                        st.subheader("检测结果")
                        
                        if prediction > 0.5:
                            diagnosis = "肺炎"
                            confidence = prediction * 100
                            color = "#f44336"
                        else:
                            diagnosis = "正常"
                            confidence = (1 - prediction) * 100
                            color = "#4caf50"
                        
                        st.markdown(f"""
                        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 10px 0;">
                            <h3 style="color: {color};">诊断结果: {diagnosis}</h3>
                            <div style="width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
                                <div style="width: {confidence}%; height: 100%; background-color: {color}; border-radius: 10px;"></div>
                            </div>
                            <p style="margin-top: 10px;">信心度: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()