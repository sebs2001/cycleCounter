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

# Camera input (mobile and desktop compatible)
uploaded_file = st.camera_input("Take a photo")

if uploaded_file:
    # Convert to base64
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    # Roboflow API endpoint
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    # Show original image
    st.image(uploaded_file, caption="Original Image")

    # Send image to Roboflow
    response = requests.post(api_url, json={"image": img_str})

    if response.status_code == 200:
        try:
            result = response.json()

            # Show annotated image if available
            if "image" in result and "url" in result["image"]:
                st.image(result["image"]["url"], caption="Roboflow Detection (Annotated)")

            # Count predictions
            if "predictions" in result:
                count = len(result["predictions"])
                st.success(f"Detected {count} resistors.")
            else:
                st.warning("No predictions returned.")
        except Exception as e:
            st.error(f"❌ Failed to decode Roboflow JSON: {e}")
            st.text(response.text)
            st.stop()
    else:
        st.error(f"❌ Roboflow API Error: {response.status_code}")
        st.text(response.text)
        st.stop()

    # Upload image to Roboflow for annotation
    if st.button("💾 Upload to Roboflow"):
        rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
        project = rf.workspace("quanticwork").project("my-first-project-eintr")

        # Save the uploaded image locally
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        try:
            project.upload(
                image_path=temp_path,
                batch_name="streamlit-submits",
                split="train",
                tag_names=["from-streamlit"],
                num_retry_uploads=3
            )
            st.success("✅ Uploaded to Roboflow. Check Annotate > Unannotated.")
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")



