import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import pandas as pd
from fpdf import FPDF
import datetime

@tf.keras.utils.register_keras_serializable()
def l2_normalization(x):
    return tf.math.l2_normalize(x, axis=-1)

custom_objects = {"l2_normalization": l2_normalization}
binary_model = load_model("model/binary_model.h5", custom_objects=custom_objects, compile=False)
benign_model = load_model("model/benign_model.h5", custom_objects=custom_objects, compile=False)
malignant_model = load_model("model/malignant_model.h5", custom_objects=custom_objects, compile=False)
grade_model = load_model("model/grade_model.h5", custom_objects=custom_objects, compile=False)

binary_labels = ["Benign", "Malignant"]
benign_labels = ["Adenosis (A)", "Fibroadenoma (F)", "Phyllodes Tumor (PT)", "Tubular Adenoma (TA)"]
malignant_labels = ["Carcinoma (DC)", "Lobular Carcinoma (LC)", "Mucinous Carcinoma (MC)", "Papillary Carcinoma (PC)"]
grade_labels = ["Grade 1", "Grade 2", "Grade 3"]

def predict_image(image_path, model, image_size=(128, 128)):
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class, prediction

def generate_pdf(name, age, sex, classification_type, result, subtype=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Breast Cancer Diagnosis Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient Name: {name}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
    pdf.cell(200, 10, txt=f"Sex: {sex}", ln=True)
    pdf.cell(200, 10, txt=f"Classification Type: {classification_type}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    if subtype:
        pdf.cell(200, 10, txt=f"Subtype: {subtype}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(report_path)
    return report_path

def store_case_data(name, age, sex, classification_type, result, subtype):
    os.makedirs("records", exist_ok=True)
    record = {
        "Name": name,
        "Age": age,
        "Sex": sex,
        "Classification Type": classification_type,
        "Result": result,
        "Subtype": subtype,
        "Timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    df = pd.DataFrame([record])
    excel_path = "records/case_data.xlsx"
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel(excel_path, index=False)
    else:
        df.to_excel(excel_path, index=False)

def main():
    st.title("🧬 Breast Cancer Detection")
    st.write("Upload a histopathology image and receive a diagnostic report.")

    name = st.text_input("Patient Name")
    age = st.text_input("Age")
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    task = st.radio("Choose Classification Type", ["Type Classification", "Grade Classification"])

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file and name and age and sex:
        image_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("🔍 Processing...")

        result = ""
        subtype = ""

        if task == "Type Classification":
            cls_idx, _ = predict_image(image_path, binary_model)
            result = binary_labels[cls_idx]
            st.success(f"Diagnosis: {result}")
            if result == "Benign":
                sub_idx, _ = predict_image(image_path, benign_model)
                subtype = benign_labels[sub_idx]
            else:
                sub_idx, _ = predict_image(image_path, malignant_model)
                subtype = malignant_labels[sub_idx]
            st.info(f"Subtype: {subtype}")

        elif task == "Grade Classification":
            cls_idx, _ = predict_image(image_path, grade_model)
            result = grade_labels[cls_idx]
            st.success(f"Grade: {result}")

        store_case_data(name, age, sex, task, result, subtype)

        report_path = generate_pdf(name, age, sex, task, result, subtype)
        with open(report_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name=os.path.basename(report_path), mime="application/pdf")

if __name__ == "__main__":
    main()
