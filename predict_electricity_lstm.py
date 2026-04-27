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
            # --- الحل الجذري هنا ---
            # بنشوف الـ Scaler محتاج كام عمود (سواء 10 أو 12 أو غيره)
            num_features_needed = scaler.n_features_in_
            
            # بنعمل مصفوفة بالأعمدة المطلوبة كلها أصفار
            data_to_scale = np.zeros((1, num_features_needed))
            data_to_scale[0, 0] = input_val  # بنحط رقمك في أول عمود
            
            # بنعمل الـ Scaling
            scaled_data = scaler.transform(data_to_scale)
            
            if "MLP" in model_choice:
                # بنقطع البيانات عشان تناسب الموديل (أول 10 أعمدة)
                # لو الموديل محتاج 10 والـ scaler مطلع 12، هناخد أول 10 بس
                mlp_input = scaled_data[:, :mlp_model.input_shape[1]]
                res = mlp_model.predict(mlp_input)[0][0]
            else:
                # لـ LSTM برضه بناخد اللي الموديل محتاجه ونغير الشكل لثلاثي الأبعاد
                lstm_features = lstm_model.input_shape[-1]
                lstm_input = scaled_data[:, :lstm_features].reshape(1, 1, lstm_features)
                res = lstm_model.predict(lstm_input)[0][0]

            # عرض النتيجة
            st.metric(label=f"التكلفة المتوقعة ({model_choice})", value=f"{res:.2f} جنيه")
            st.balloons()
            
        except Exception as e:
            st.error(f"حدث خطأ تقني: {e}")
            st.write("نصيحة: تأكدي أن ملفات الموديلات هي النسخ الأخيرة.")
    else:
        st.write("أدخل البيانات ثم اضغط على الزر...")

st.markdown("---")
st.caption( "مشروع التخرج 2026")
