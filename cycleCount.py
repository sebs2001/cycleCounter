import streamlit as st
import requests
import base64
import os
from roboflow import Roboflow

# Make sure save folder exists
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Camera input (mobile and desktop compatible)
uploaded_file = st.camera_input("Take a photo")

if uploaded_file:
    # Convert to base64
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")
    img_str = f"data:image/jpeg;base64,{img_str}"  # ✅ Required prefix for Roboflow API

    # Roboflow Hosted Inference API
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    try:
        response = requests.post(api_url, json={"image": img_str})
        result = response.json()

        # Show original image
        st.image(uploaded_file, caption="Original Image")

        # Show annotated image
        if "image" in result and "url" in result["image"]:
            st.image(result["image"]["url"], caption="Roboflow Detection")

        # Show count
        if "predictions" in result:
            count = len(result["predictions"])
            st.success(f"Detected {count} resistors.")
        else:
            st.warning("No predictions returned.")

        # Optional: save annotated image
        if st.button("💾 Save annotated image"):
            if "image" in result and "url" in result["image"]:
                image_data = requests.get(result["image"]["url"]).content
                image_id = len(os.listdir("saved"))
                with open(f"saved/resistor_{image_id}.jpg", "wb") as f:
                    f.write(image_data)
                st.success("Annotated image saved.")

        # Upload to Roboflow for re-annotation
        if st.button("📤 Upload to Roboflow"):
            try:
                rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
                project = rf.workspace("quanticwork").project("my-first-project-eintr")

                # Save temporarily
                temp_path = "temp_upload.jpg"
                with open(temp_path, "wb") as f:
                    f.write(img_bytes)

                upload_response = project.upload(
                    image_path=temp_path,
                    batch_name="streamlit-submits",
                    split="train",
                    num_retry_uploads=3,
                    tag_names=["from-streamlit"]
                )

                st.success("✅ Uploaded to Roboflow. Check Annotate → Unannotated.")

            except Exception as e:
                st.error(f"❌ Upload failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    st.info("📷 Please take a photo to begin.")










