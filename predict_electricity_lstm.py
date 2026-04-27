import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# 1. إعدادات الصفحة
st.set_page_config(page_title="نظام التوقع الذكي"، layout="wide")

@st.cache_resource
def load_all():
    scaler = joblib.load('scaler.joblib')
    # تحميل الموديل بدون الـ compilation عشان نتفادى مشاكل الـ Custom Objects
    lstm = tf.keras.models.load_model('multi_output_lstm_model.h5', compile=False)
    return scaler, lstm

try:
    scaler, lstm_model = load_all()
    st.sidebar.success("✅ الأنظمة جاهزة")
except Exception as e:
    st.sidebar.error("❌ تأكدي من وجود الملفات")

st.title("⚡ لوحة تحكم المدينة الذكية - توقع التكلفة")
st.markdown("---")

# 2. واجهة المستخدم (الـ 12 عمود بتوعك)
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
        # تجهيز الداتا (12 مدخل + صفر للسعر عشان يكملوا 13)
        input_raw = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, 0.0]])
        
        # تحويل البيانات بالميزان
        scaled_data = scaler.transform(input_raw)
        
        # التشكيل للـ LSTM
        lstm_input = scaled_data.reshape(1, 1, 13)
        
        # التوقع
        prediction = lstm_model.predict(lstm_input)
        
        # هنا الخلاصة: بما إنه Multi-output، الموديل بيطلع 13 قيمة
        # إحنا بنعمل Inverse Transform عشان نرجع الأرقام لأصلها (دولار/جنيه)
        # وبناخد القيمة رقم 13 (index -1) اللي هي التكلفة
        unscaled_prediction = scaler.inverse_transform(prediction.reshape(1, 13))
        final_cost = unscaled_prediction[0][-1]
        
        st.balloons()
        st.success(f"### 💰 التكلفة المتوقعة: {abs(final_cost):,.2f} دولار")
        
        # عرض مقارنة بسيطة (اختياري للمناقشة)
        st.info(f"💡 تم الحساب بناءً على استهلاك {f12} kWh وعوامل البيئة المحيطة.")
        
    except Exception as e:
        st.error(f"حدث خطأ: {e}")

st.markdown("---")
st.caption("مشروع التخرج 2026 - المهندسة سلمى")
