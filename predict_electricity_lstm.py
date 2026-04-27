import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# 1. إعدادات الصفحة
st.set_page_config(page_title="توقع الطاقة الذكي", page_icon="⚡", layout="wide")

# 2. وظيفة تحميل الموديلات
@st.cache_resource
def load_all_assets():
    mlp = joblib.load('mlp_doubled_neurons_model.joblib')
    scaler = joblib.load('scaler.joblib')
    lstm = tf.keras.models.load_model('multi_output_lstm_model.h5', compile=False)
    return mlp, scaler, lstm

try:
    mlp_model, scaler, lstm_model = load_all_assets()
    st.sidebar.success("✅ الأنظمة جاهزة")
except Exception as e:
    st.sidebar.error(f"❌ مشكلة في تحميل الملفات: {e}")

# 3. واجهة المستخدم
st.title("🤖 نظام توقع استهلاك الطاقة الكهربائية")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.info("### ⚙️ الإعدادات")
    # اختيار الموديل
    model_choice = st.radio(
        "اختر تقنية الذكاء الاصطناعي:",
        ("الموديل الأساسي (MLP)", "الموديل المتطور (LSTM)")
    )
    # إدخال القيمة
    input_val = st.number_input("كمية الاستهلاك (kWh):", min_value=0.0, value=100.0)
    predict_btn = st.button("🚀 تحليل وتوقع النتيجة")

with col2:
    st.success("### 📊 نتائج التحليل")
    if predict_btn:
        try:
            # تجهيز مصفوفة بها 10 خانات كما يتوقع الـ Scaler
            data_for_scaler = np.zeros((1, 10))
            data_for_scaler[0, 0] = input_val
            
            if "MLP" in model_choice:
                # 1. استخدام الميزان (Scaling)
                scaled_data = scaler.transform(data_for_scaler)
                # 2. التوقع باستخدام الموديل الأول
                prediction = mlp_model.predict(scaled_data)
                res = prediction[0][0]
            else:
                # موديل LSTM يتوقع بيانات ثلاثية الأبعاد (1, 1, 10)
                lstm_data = data_for_scaler.reshape(1, 1, 10)
                prediction = lstm_model.predict(lstm_data)
                res = prediction[0][0]

            # عرض النتيجة بشكل شيك
            st.metric(label=f"التكلفة المتوقعة ({model_choice})", value=f"{res:.2f} جنيه")
            st.balloons()
            
        except Exception as e:
            st.error(f"حدث خطأ في الحساب: {e}")
    else:
        st.write("انتظار إدخال البيانات والضغط على الزر...")

st.markdown("---")
st.caption(" - مشروع التخرج 2026")
