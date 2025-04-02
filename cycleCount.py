import streamlit as st
import requests
import base64
import os
from roboflow import Roboflow

# Ensure a local save folder exists
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo to count resistors.")

# Capture photo
uploaded_file = st.camera_input("📸 Capture a photo")

if uploaded_file:
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    # Roboflow API endpoint
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    # Send to Roboflow
    response = requests.post(api_url, json={"image": img_str})

    # Process only the image and count
    try:
        result = response.json()
        image_url = result["image"]["url"]
        count = len(result["predictions"])
    except:
        st.error("❌ Could not process Roboflow response.")
        st.text(response.text)
        st.stop()

    # Show original and annotated image
    st.image(uploaded_file, caption="Original Image")
    st.image(image_url, caption="Roboflow Detection (Annotated)")

    # Show detected count
    st.success(f"Detected {count} resistors.")

    # Upload to Roboflow dataset for annotation
    if st.button("💾 Upload to Roboflow"):
        try:
            rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
            project = rf.workspace("quanticwork").project("my-first-project-eintr")

            # Save photo locally to upload
            temp_path = "temp_upload.jpg"
            with open(temp_path, "wb") as f:
                f.write(img_bytes)

            project.upload(
                image_path=temp_path,
                batch_name="streamlit-submits",
                split="train",
                tag_names=["from-streamlit"],
                num_retry_uploads=3
            )

            st.success("✅ Uploaded to Roboflow. Check Annotate → Unannotated.")

        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")

