import streamlit as st
import requests
import os
import Roboflow


# Make sure save folder exists
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Camera input (mobile and desktop compatible)
uploaded_file = st.camera_input("Take a photo", key="camera1")

if uploaded_file is not None:
    # Get image bytes
    img_bytes = uploaded_file.getvalue()

    # ✅ Correct Roboflow API URL
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    # Send the image using files=, not JSON
    response = requests.post(api_url, files={"file": img_bytes})

    if response.status_code != 200:
        st.error(f"API Error: {response.status_code}")
        st.stop()

    result = response.json()

    # Debug: show the raw JSON response
    st.json(result)

    # Show the original photo
    st.image(uploaded_file, caption="Original Image")

    # Show Roboflow annotated result
    if "image" in result and "url" in result["image"]:
        st.image(result["image"]["url"], caption="Roboflow Detection")

    # Show count
    if "predictions" in result:
        count = len(result["predictions"])
        st.success(f"Detected {count} resistors.")
    else:
        st.warning("No predictions returned.")

# Optionally save the annotated image
if st.button("💾 Upload to Roboflow"):
    rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
    project = rf.workspace("quanticwork").project("my-first-project-eintr")

    if "image" in result and "url" in result["image"]:
        # Download the image from Roboflow inference result
        image_data = requests.get(result["image"]["url"]).content

        # Save it temporarily
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)

        # ✅ Upload using Roboflow SDK — official method
        upload_response = project.upload("temp_upload.jpg")

        # ✅ Show exact response so you know it worked
        st.write(upload_response.json())

        if upload_response.status_code == 200 or "id" in upload_response.json():
            st.success("✅ Uploaded. Go to Annotate → Unannotated in Roboflow.")
        else:
            st.error("❌ Upload failed.")
    else:
        st.warning("⚠️ No image found to upload.")





