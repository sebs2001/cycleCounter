import streamlit as st
import requests
import os
from roboflow import Roboflow

# Make sure save folder exists
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Camera input (mobile and desktop compatible)
uploaded_file = st.camera_input("Take a photo", key="camera1")

if uploaded_file is not None:
    # Convert uploaded image to bytes
    img_bytes = uploaded_file.getvalue()

    # Your Roboflow inference endpoint (make sure version & key match your project)
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    #### 1) First request for JSON predictions ####
    response_json = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "json"}  # request JSON so we can parse predictions
    )

    if response_json.status_code != 200:
        st.error(f"API Error (JSON): {response_json.status_code}")
        st.stop()

    result = response_json.json()
    # Debug: show the raw JSON response
    st.subheader("Raw JSON Predictions")
    st.json(result)

    # Show the original photo
    st.subheader("Original Image")
    st.image(uploaded_file, caption="Original Image")

    #### 2) Second request for the annotated/bounding-box image ####
    response_image = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "image", "annotated": "true"}  # request an annotated image
    )

    if response_image.status_code != 200:
        st.error(f"API Error (Annotated Image): {response_image.status_code}")
        st.stop()

    # Show the annotated image from Roboflow
    st.subheader("Roboflow Detection (with bounding boxes)")
    st.image(response_image.content, caption="Annotated Image from Roboflow")

    # Count the predictions (e.g. resistors)
    if "predictions" in result:
        count = len(result["predictions"])
        st.success(f"Detected {count} resistors.")
    else:
        st.warning("No predictions returned.")

# Upload to Roboflow button
if st.button("💾 Upload to Roboflow"):
    # Initialize Roboflow
    rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
    project = rf.workspace("quanticwork").project("my-first-project-eintr")

    if uploaded_file:
        # Save uploaded image locally
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Upload using Roboflow SDK with optional params
            upload_response = project.upload(
                image_path=temp_path,
                batch_name="streamlit-submits",
                split="train",
                tag_names=["from-streamlit"],
                num_retry_uploads=3
            )
            # If we reach here, upload worked
            st.success("✅ Uploaded to Roboflow. Check Annotate > Unannotated.")
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
    else:
        st.warning("⚠️ No image available to upload.")
