import streamlit as st
import requests
import os
from roboflow import Roboflow

# Ensure 'saved' directory exists
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Camera input (mobile and desktop compatible)
uploaded_file = st.camera_input("Take a photo", key="camera1")

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()

    # Roboflow Inference API
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8"
    params = {
        "api_key": "o9tbMpy3YklEF3MoRmdR",
        "format": "json",        # <- Make sure this is JSON
        "annotated": "true"
    }

    # Send image to Roboflow
    response = requests.post(api_url, files={"file": img_bytes}, params=params)

    if response.status_code == 200:
        try:
            result = response.json()

            # Show original photo
            st.image(uploaded_file, caption="Original Image")

            # Show annotated image (bounding boxes from model)
            image_url = result.get("image", {}).get("url")
            if image_url:
                st.image(image_url, caption="Roboflow Detection (Annotated)")
            else:
                st.warning("⚠️ Annotated image not returned. Check model output settings.")

            # Show number of detected resistors
            predictions = result.get("predictions", [])
            st.success(f"✅ Detected {len(predictions)} resistors.")

        except Exception as e:
            st.error(f"❌ Failed to parse JSON result: {str(e)}")
    else:
        st.error(f"❌ Roboflow API request failed (Status {response.status_code})")

# Upload to Roboflow for annotation
if st.button("💾 Upload to Roboflow"):
    if uploaded_file:
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
            project = rf.workspace("quanticwork").project("my-first-project-eintr")
            project.upload(
                image_path=temp_path,
                batch_name="streamlit-submits",
                split="train",
                tag_names=["from-streamlit"],
                num_retry_uploads=3
            )
            st.success("📤 Uploaded to Roboflow. Check Annotate > Unannotated.")
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
    else:
        st.warning("⚠️ No image to upload.")
