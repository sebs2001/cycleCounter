import os
import requests
import streamlit as st
from roboflow import Roboflow

# Create "saved" folder if it doesn't exist
os.makedirs("saved", exist_ok=True)

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Camera input widget (mobile and desktop)
uploaded_file = st.camera_input("Take a photo", key="camera1")

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    # Request and display the annotated image (binary response)
    response_image = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "image", "annotated": "true"}
    )
    if response_image.status_code == 200:
        st.image(response_image.content, caption="Annotated Image with bounding boxes")
    else:
        st.error(f"Error fetching annotated image: {response_image.status_code}")

    # Request JSON predictions for counting
    response_json = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "json"}
    )
    if response_json.status_code == 200:
        predictions = response_json.json().get("predictions", [])
        if predictions:
            st.success(f"Detected {len(predictions)} resistors.")
        else:
            st.warning("No predictions returned.")
    else:
        st.error(f"Error fetching predictions: {response_json.status_code}")

# UPLOAD SECTION: Upload image using Roboflow Python client
if st.button("💾 Upload to Roboflow"):
    if not uploaded_file:
        st.warning("⚠️ No image available to upload.")
    else:
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        try:
            rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
            project = rf.workspace("quanticwork").project("my-first-project-eintr")
            upload_response = project.upload(temp_path)
            st.success("✅ Uploaded to Roboflow. Check 'Annotate > Unannotated' in your project.")
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")



