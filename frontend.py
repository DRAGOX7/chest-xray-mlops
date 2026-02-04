import streamlit as st
import requests
from PIL import Image

# --- CONFIGURATION ---
# This is the address of your Azure backend
API_URL = "https://aljaf-xray-final.azurewebsites.net/predict"

st.set_page_config(page_title="Abdullah AI Diagnostic", page_icon="🩻")

# --- UI HEADER ---
st.title("🩻 Abdullah AI - Chest X-Ray Diagnostic")
st.markdown("Upload a chest X-ray image to detect abnormalities using **DenseNet121**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("About")
    st.write("This AI model is hosted on Microsoft Azure.")
    st.write("Built with PyTorch & FastAPI.")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Show the image to the user
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-Ray", use_column_width=True)

    # 2. Add a button to predict
    if st.button("Analyze Image"):
        with st.spinner("Consulting the AI Doctor..."):
            try:
                # Prepare the file for the API
                # We need to reset the pointer to the start of the file
                uploaded_file.seek(0)
                files = {"file": uploaded_file}

                # Send request to Azure
                response = requests.post(API_URL, files=files)

                # Get the result
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    diagnosis = result['diagnosis']
                    prob = float(result['abnormality_probability'])
                    
                    st.divider()
                    
                    if diagnosis == "Abnormal":
                        st.error(f"🚨 **Diagnosis: ABNORMAL**")
                    else:
                        st.success(f"✅ **Diagnosis: NORMAL**")
                        
                    st.write(f"**Confidence:** {prob*100:.2f}%")
                    st.json(result) # Show raw data
                else:
                    st.error("Error communicating with the backend.")
                    
            except Exception as e:
                st.error(f"Connection failed: {e}")