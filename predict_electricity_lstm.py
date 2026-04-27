import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="توقع تكلفة الكهرباء - 12 Feature", layout="wide")

@st.cache_resource
def load_all():
    scaler = joblib.load('scaler.joblib')
    lstm = tf.keras.models.load_model('multi_output_lstm_model.h5', compile=False)
    return scaler, lstm

scaler, lstm_model = load_all()

st.title("⚡ نظام إدارة طاقة المدينة الذكية")
st.markdown("---")

# 12 مدخل بالظبط زي ترتيب الإكسيل بتاعك
col1, col2, col3 = st.columns(3)

with col1:
    f1 = st.number_input("Site Area:", value=2000.0)
    f2 = st.number_input("Water Consumption:", value=4000.0)
    f3 = st.number_input("Recycling Rate:", value=20.0)
    f4 = st.number_input("Utilisation Rate:", value=50.0)

with col2:
    f5 = st.number_input("Air Quality Index:", value=70.0)
    f6 = st.number_input("Issue Resolution:", value=60.0)
    f7 = st.number_input("Resident Satisfaction:", value=80.0)
    f8 = st.number_input("Carbon Emissions:", value=100.0)

with col3:
    f9 = st.selectbox("Structure Type 1:", [0, 1])
    f10 = st.selectbox("Structure Type 2:", [0, 1])
    f11 = st.selectbox("Structure Type 3:", [0, 1])
    f12 = st.number_input("Electricity Consumption:", value=1500.0)

if st.button("🚀 توقع التكلفة الآن"):
    try:
        # 1. بنجمع الـ 12 مدخل اللي دخلتيهم
        user_inputs = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12]
        
        # 2. التركة: بنزود رقم "صفر" يدوي عشان نكملهم 13 خانة للميزان
        full_input = np.array([user_inputs + [0.0]]) 
        
        # 3. الميزان (Scaler) هيشتغل دلوقتي صح لأن لقى 13 خانة
        scaled_data = scaler.transform(full_input)
        
        # 4. التوقع (LSTM) - بنشكله (1, 1, 13) زي ما طلب في الصورة
        lstm_input = scaled_data.reshape(1, 1, 13)
        
        prediction = lstm_model.predict(lstm_input)
        res = prediction[0][0]
        
        st.success(f"### التكلفة المتوقعة: {abs(res):.2f} دولار")
        st.balloons()
        
    except Exception as e:
        st.error(f"الخطأ لسه موجود: {e}")
