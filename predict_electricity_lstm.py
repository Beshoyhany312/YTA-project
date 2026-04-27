import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# 1. إعداد واجهة الموقع
st.set_page_config(page_title="نظام التوقع المزدوج - سلمى", page_icon="⚡")

# 2. وظيفة تحميل الموديلات
@st.cache_resource
def load_all_assets():
    # الموديل الأول والـ Scaler
    mlp = joblib.load('mlp_doubled_neurons_model.joblib')
    scaler = joblib.load('scaler.joblib')
    # الموديل الثاني (LSTM)
    lstm = tf.keras.models.load_model('multi_output_lstm_model.h5',compile=false)
    return mlp, scaler, lstm

try:
    mlp_model, scaler, lstm_model = load_all_assets()
    st.sidebar.success("✅ جميع الموديلات جاهزة")
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل الملفات: {e}")

# 3. تصميم الصفحة
st.title("⚡ نظام توقع فاتورة الكهرباء الذكي")
st.write("أهلاً يا سلمى! اختاري الموديل وقارني بين النتائج")

# قائمة اختيار الموديل في الجانب
model_choice = st.sidebar.radio(
    "اختر نوع الموديل:",
    ("الموديل الأساسي (MLP)", "الموديل المتطور (LSTM)")
)

# خانة إدخال الاستهلاك
input_val = st.number_input("أدخل كمية الاستهلاك (kWh):", min_value=0.0, value=100.0)

if st.button("بدء التوقع"):
    try:
        if model_choice == "الموديل الأساسي (MLP)":
            # تجهيز البيانات للموديل الأول
            data = np.zeros((1, 10))
            data[0, 0] = input_val
            scaled_data = scaler.transform(data)
            prediction = mlp_model.predict(scaled_data)
            final_result = prediction[0][0]
        else:
            # تجهيز البيانات للموديل الثاني
            data = np.zeros((1, 1, 10))
            data[0, 0, 0] = input_val
            prediction = lstm_model.predict(data)
            final_result = prediction[0][0]

        # عرض النتيجة
        st.markdown(f"### النتيجة باستخدام {model_choice}:")
        st.info(f"التكلفة المتوقعة هي: **{final_result:.4f}**")
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء الحساب: {e}")

st.markdown("---")
st.caption("تم التطوير بواسطة سلمى - مشروع التخرج 2026")
