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
    st.sidebar.error("❌ مشكلة في الملفات")

# 3. واجهة المستخدم
st.title("🤖 نظام توقع استهلاك الطاقة الكهربائية")
st.markdown("---")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.info("### ⚙️ الإعدادات")
    model_choice = st.radio("اختر تقنية الذكاء الاصطناعي:", ("الموديل الأساسي (MLP)", "الموديل المتطور (LSTM)"))
    input_val = st.number_input("كمية الاستهلاك (kWh):", min_value=0.0, value=100.0)
    predict_btn = st.button("🚀 تحليل وتوقع النتيجة")

with col2:
    st.success("### 📊 نتائج التحليل")
    if predict_btn:
        try:
            # الخطوة 1: نعرف الـ Scaler محتاج كام عمود ونجهزهم
            n_scaler = scaler.n_features_in_
            data_for_sc = np.zeros((1, n_scaler))
            data_for_sc[0, 0] = input_val
            
            # الخطوة 2: نعمل الـ Scaling
            scaled_full = scaler.transform(data_for_sc)
            
            if "MLP" in model_choice:
                # الخطوة 3: نعرف الـ MLP محتاج كام عمود بالظبط (هنا السر!)
                # بنجرب نقرأها من الموديل مباشرة
                try:
                    n_mlp = mlp_model.n_features_in_
                except:
                    n_mlp = mlp_model.coefs_[0].shape[0]
                
                final_input = scaled_full[:, :n_mlp]
                res = mlp_model.predict(final_input)[0][0]
            else:
                # الخطوة 4: للـ LSTM برضه نعرف محتاج كام
                n_lstm = lstm_model.input_shape[-1]
                final_input = scaled_full[:, :n_lstm].reshape(1, 1, n_lstm)
                res = lstm_model.predict(final_input)[0][0]

            st.metric(label=f"التكلفة المتوقعة ({model_choice})", value=f"{abs(res):.2f} جنيه")
            st.balloons()
            
        except Exception as e:
            st.error(f"خطأ في الحساب: {e}")
            st.write("حاولي التأكد من أن قيم المدخلات منطقية.")
    else:
        st.write("أدخل البيانات ثم اضغط على الزر...")

st.markdown("---")
st.caption(" مشروع التخرج 2026")
