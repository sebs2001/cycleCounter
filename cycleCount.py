import streamlit as st
import requests
import base64
import os

# Make sure save folder exists
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Camera input (mobile and desktop compatible)
uploaded_file = st.camera_input("Take a photo")

uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    api_url = "https://detect.roboflow.com/my-first-project/8?api_key=o9tbMpy3YklEF3MoRmdR"
    response = requests.post(api_url, json={"image": img_str})
    result = response.json()

    # ✅ This line is safe now — inside the block
    st.image(uploaded_file, caption="Original Image")

    if "image" in result and "url" in result["image"]:
        st.image(result["image"]["url"], caption="Roboflow Detection")

    if "predictions" in result:
        count = len(result["predictions"])
        st.success(f"Detected {count} resistors.")
    else:
        st.warning("No predictions returned.")


    # Optional: save annotated image
   if st.button("💾 Upload to Roboflow"):
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

            st.success("✅ Uploaded to Roboflow. Check Annotate > Unannotated.")
            st.write(upload_response.json())  # Show details

        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
    else:
        st.warning("⚠️ No image available to upload.")






