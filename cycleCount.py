import streamlit as st
import requests
import base64
import os
from roboflow import Roboflow

# Ensure 'saved' folder exists
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo of the component strip below.")

# Capture image from camera
uploaded_file = st.camera_input("Take a photo")

if uploaded_file:
    # Convert image to base64 string
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    # Roboflow inference endpoint
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"


    # Send request to Roboflow
    response = requests.post(api_url, json={"image": img_str})
    result = response.json()
    st.json(result)  # This will show the full raw API response


    # Show the original uploaded image
    st.image(uploaded_file, caption="Original Image")

    # Count and display number of detections
    if "predictions" in result:
        count = len(result["predictions"])
        st.success(f"Detected {count} resistors.")
    else:
        st.warning("No predictions returned.")

    # Upload to Roboflow for annotation
    if st.button("💾 Upload to Roboflow"):
        rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
        project = rf.workspace("quanticwork").project("my-first-project-eintr")

        # Save the image temporarily
        with open("temp.jpg", "wb") as f:
            f.write(img_bytes)

        # Upload to Roboflow project
        project.upload(
            image_path="temp.jpg",
            batch_name="streamlit-submits",
            split="train",
            num_retry_uploads=3,
            tag_names=["from-streamlit"]
        )

        st.success("✅ Uploaded to Roboflow. Check Annotate > Unannotated.")








