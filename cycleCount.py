import streamlit as st
import requests
import os

# Make sure the Roboflow Python client is installed and upgraded:
#   pip install --upgrade roboflow
from roboflow import Roboflow

# Ensure a folder exists for saving images if needed
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Camera input widget (works on both mobile and desktop)
uploaded_file = st.camera_input("Take a photo", key="camera1")

if uploaded_file is not None:
    # Get image bytes from the uploaded file
    img_bytes = uploaded_file.getvalue()

    # -- Roboflow Inference Endpoint (update with your correct version/API key) --
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    # 1) Request the annotated image (binary)
    response_image = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "image", "annotated": "true"}
    )

    # Show the annotated image with bounding boxes
    if response_image.status_code == 200:
        st.image(response_image.content, caption="Annotated Image with bounding boxes")
    else:
        st.error(f"Error fetching annotated image: {response_image.status_code}")

    # 2) Request JSON predictions (for counting)
    response_json = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "json"}  # annotated=true not strictly needed here
    )

    if response_json.status_code == 200:
        result = response_json.json()
        if "predictions" in result:
            count = len(result["predictions"])
            st.success(f"Detected {count} resistors.")
        else:
            st.warning("No predictions returned.")
    else:
        st.error(f"Error fetching predictions: {response_json.status_code}")


# --------------------------
# UPLOAD SECTION: Use Roboflow Python client so images appear in "Unannotated"
# --------------------------
if st.button("💾 Upload to Roboflow"):
    if uploaded_file:
        # Save the uploaded image temporarily
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # 1) Initialize Roboflow
            rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")

            # 2) Reference your workspace & project (exact names from Roboflow)
            project = rf.workspace("quanticwork").project("my-first-project-eintr")

            # 3) Minimal upload call
            #    (If you need advanced params, see below)
            upload_response = project.upload(temp_path)

            # If you need advanced params and your plan supports them, try:
            # upload_response = project.upload(
            #     image_path=temp_path,
            #     split="train",
            #     batch_name="streamlit-submits",
            #     tag_names=["from-streamlit"],
            #     num_retry_uploads=3
            # )

            st.success("✅ Uploaded to Roboflow. Check 'Annotate > Unannotated' in your project.")

            # Optionally, display the library's return value:
            # st.write(upload_response)

        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
    else:
        st.warning("⚠️ No image available to upload.")


