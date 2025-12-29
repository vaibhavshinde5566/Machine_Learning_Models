import streamlit as st
import pickle
import numpy as np
import os

# Set page configuration
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Beautiful" Dark Design
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input {
        background-color: #334155;
        color: white;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    .stNumberInput label {
        color: #94a3b8 !important;
    }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.6rem 2.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
        background: linear-gradient(90deg, #2563eb, #7c3aed);
    }
    
    /* Result Card */
    .result-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        animation: fadeIn 0.8s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-title {
        color: #cbd5e1;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    
    .result-value {
        background: -webkit-linear-gradient(#4ade80, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Input Container */
    .input-container {
        background: #1e293b;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Main App Container
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>💰 Salary Prediction AI</h1>", unsafe_allow_html=True)

# Load Model
model_path = 'salary_model.pkl'

@st.cache_resource
def load_model():
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"⚠️ Model file '{model_path}' not found! Please ensure it exists in the directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_model()

if model:
    # Input Section
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown("### 💼 Professional Details")
    
    experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=2.5,
        step=0.1,
        help="Enter your total years of professional experience."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Button
    if st.button("🚀 Predict Salary"):
        # Make prediction
        input_data = np.array([[experience]]).reshape(-1, 1)
        
        try:
            prediction = model.predict(input_data)
            predicted_salary = prediction[0]
            
            # Display Result
            st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">Estimated Annual Salary</div>
                    <div class="result-value">${predicted_salary:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.warning("⚠️ Model file missing. Please regenerate the model.")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 60px; color: #64748b; font-size: 0.8rem;">
        Designed for Excellence • SalaryAI v1.0
    </div>
""", unsafe_allow_html=True)
