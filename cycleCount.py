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

    # Your Roboflow inference endpoint (replace with your project’s version & key)
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    # Request JSON + annotated image
    response = requests.post(
        api_url,
        files={"file": img_bytes},
        params={
            "format": "json",       # get predictions as JSON
            "annotated": "true"     # also generate bounding-box image
        }
    )

    if response.status_code != 200:
        st.error(f"API Error: {response.status_code}")
        st.stop()

    # Parse JSON
    result = response.json()

    # Debug: show the raw JSON response
    st.subheader("Raw JSON Predictions")
    st.json(result)

    # Show the original photo
    st.subheader("Original Image")
    st.image(uploaded_file, caption="Original Image")

    # Show the annotated image from Roboflow
    if "image" in result and "url" in result["image"]:
        st.subheader("Annotated Image from Roboflow")
        st.image(result["image"]["url"], caption="Roboflow Detection (with bounding boxes)")
    else:
        st.warning("No annotated image URL found in the JSON response.")

    # Show the count of predictions
    if "predictions" in result:
        count = len(result["predictions"])
        st.success(f"Detected {count} resistors.")
    else:
        st.warning("No predictions returned.")

# Button to upload the image to Roboflow (for labeling/training)
if st.button("💾 Upload to Roboflow"):
    rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
    project = rf.workspace("quanticwork").project("my-first-project-eintr")

    if uploaded_file:
        # Save the uploaded image locally
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Upload using Roboflow’s Python SDK
            upload_response = project.upload(
                image_path=temp_path,
                batch_name="streamlit-submits",
                split="train",
                tag_names=["from-streamlit"],
                num_retry_uploads=3
            )
            st.success("✅ Uploaded to Roboflow. Check 'Annotate' > 'Unannotated' in your project.")
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
    else:
        st.warning("⚠️ No image available to upload.")
