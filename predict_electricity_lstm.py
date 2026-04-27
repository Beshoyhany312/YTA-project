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
    st.sidebar.error("❌ مشكلة في تحميل الملفات")

# 3. واجهة المستخدم
st.title("🤖 نظام توقع استهلاك الطاقة الكهربائية")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.info("### ⚙️ الإعدادات")
    model_choice = st.radio(
        "اختر تقنية الذكاء الاصطناعي:",
        ("الموديل الأساسي (MLP)", "الموديل المتطور (LSTM)")
    )
    input_val = st.number_input("كمية الاستهلاك (kWh):", min_value=0.0, value=100.0)
    predict_btn = st.button("🚀 تحليل وتوقع النتيجة")

with col2:
    st.success("### 📊 نتائج التحليل")
    if predict_btn:
        try:
            # الحل الجذري للاختلاف بين الـ Scaler والموديل
            # 1. بنشوف الـ Scaler محتاج كام (غالباً 12)
            num_scaler_features = scaler.n_features_in_
            data_for_scaler = np.zeros((1, num_scaler_features))
            data_for_scaler[0, 0] = input_val
            
            # 2. بنعمل الـ Scaling
            scaled_full = scaler.transform(data_for_scaler)
            
            if "MLP" in model_choice:
                # الموديل محتاج أول 10 أعمدة بس من الـ 12
                final_input = scaled_full[:, :10]
                res = mlp_model.predict(final_input)[0][0]
            else:
                # موديل LSTM برضه بياخد أول 10 أعمدة وشكل ثلاثي الأبعاد
                final_input = scaled_full[:, :10].reshape(1, 1, 10)
                res = lstm_model.predict(final_input)[0][0]

            # عرض النتيجة
            st.metric(label=f"التكلفة المتوقعة ({model_choice})", value=f"{res:.2f} جنيه")
            st.balloons()
            
        except Exception as e:
            st.error(f"حدث خطأ في الحساب: {e}")
    else:
        st.write("أدخل البيانات ثم اضغط على الزر...")

st.markdown("---")
st.caption(" مشروع التخرج 2026")
