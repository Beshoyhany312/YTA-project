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
            # الحل السحري: نعرف الميزان محتاج كام عمود بالظبط ونملاهم أصفار
            num_features = scaler.n_features_in_
            data_ready = np.zeros((1, num_features))
            data_ready[0, 0] = input_val # بنحط استهلاكك في أول خانة
            
            if "MLP" in model_choice:
                # التحويل باستخدام الميزان (Scaling)
                scaled_data = scaler.transform(data_ready)
                # التوقع
                res = mlp_model.predict(scaled_data)[0][0]
            else:
                # لـ LSTM بنحتاج نعدل شكل البيانات (1, 1, عدد الأعمدة)
                # لو الـ LSTM متدرب على 10 بس، هنعدل الـ data_ready
                try:
                    lstm_input = data_ready[:, :10].reshape(1, 1, 10)
                    res = lstm_model.predict(lstm_input)[0][0]
                except:
                    lstm_input = data_ready.reshape(1, 1, num_features)
                    res = lstm_model.predict(lstm_input)[0][0]

            # عرض النتيجة بشكل شيك جداً
            st.metric(label=f"التكلفة المتوقعة ({model_choice})", value=f"{res:.2f} جنيه")
            st.balloons()
            
        except Exception as e:
            st.error(f"حدث خطأ في الحساب: {e}")
    else:
        st.write("أدخل البيانات ثم اضغط على الزر ليظهر التوقع هنا.")

st.markdown("---")
st.caption(" مشروع التخرج 2026")
