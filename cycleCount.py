import streamlit as st
import requests
import base64
import os
from roboflow import Roboflow

# --- CONFIGURATION ---
API_KEY = "o9tbMpy3YklEF3MoRmdR"
MODEL_ENDPOINT = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"
WORKSPACE = "quanticwork"
PROJECT_ID = "my-first-project-eintr"

# --- CREATE SAVE FOLDER IF NOT EXISTS ---
if not os.path.exists("saved"):
    os.makedirs("saved")

# --- UI HEADER ---
st.title("Cycle Counter v1.0")
st.write("📸 Take a photo of the resistor strip.")

# --- CAMERA INPUT ---
uploaded_file = st.camera_input("Take a photo")

if uploaded_file:
    # --- SHOW ORIGINAL IMAGE ---
    st.image(uploaded_file, caption="Original Image")

    # --- Convert to base64 for Roboflow detection API ---
    img_bytes = uploaded_file.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # --- Send to Roboflow hosted inference ---
    response = requests.post(MODEL_ENDPOINT, json={"image": img_b64})

    if response.status_code == 200:
        result = response.json()

        # --- Show annotated image (if present) ---
        if "image" in result and "url" in result["image"]:
            st.image(result["image"]["url"], caption="Detected Resistors")

        # --- Count detections ---
        count = len(result.get("predictions", []))
        st.success(f"✅ Detected {count} resistors.")
    else:
        st.error(f"❌ Detection failed. Status: {response.status_code}")
        st.stop()

    # --- UPLOAD TO ANNOTATION ONLY IF BUTTON PRESSED ---
    if st.button("💾 Upload to Roboflow for Annotation"):
        # Save image temporarily
        temp_path = "UPLOAD_IMAGE.jpg"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        # Roboflow SDK Upload (as documented)
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT_ID)

        try:
            project.upload(
                image_path=temp_path,
                batch_name="streamlit-submits",
                split="train",
                num_retry_uploads=3,
                tag_names=["from-streamlit"],
                sequence_number=99,
                sequence_size=100
            )
            st.success("🟢 Uploaded to Roboflow: Check Annotate > Unannotated.")
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")




