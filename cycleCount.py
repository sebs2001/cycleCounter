import streamlit as st
import requests
import base64
import os
from roboflow import Roboflow

# Create a folder for local saves (if needed)
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Take photo input
uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    # Convert photo to base64
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    # Roboflow hosted model API
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"
    response = requests.post(api_url, json={"image": img_str})
    
    try:
        result = response.json()
    except Exception:
        st.error("❌ Error decoding Roboflow response.")
        st.stop()

    # Show original image
    st.image(uploaded_file, caption="Original Image")

    # Show annotated detection
    if "image" in result and "url" in result["image"]:
        st.image(result["image"]["url"], caption="Roboflow Detection")

    # Count detected components
    if "predictions" in result:
        count = len(result["predictions"])
        st.success(f"Detected {count} resistors.")
    else:
        st.warning("No predictions returned.")

    # Upload to Roboflow Dataset
    if st.button("💾 Upload to Roboflow"):
        rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
        project = rf.workspace("quanticwork").project("my-first-project-eintr")

        # Save photo locally before upload
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        try:
            upload_response = project.upload(
                image_path=temp_path,
                batch_name="streamlit-submits",
                split="train",
                tag_names=["from-streamlit"],
                num_retry_uploads=3
            )
            st.success("✅ Uploaded to Roboflow. Check Annotate > Unannotated.")
        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")








