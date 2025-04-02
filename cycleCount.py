import streamlit as st
import requests
import os


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
    response = requests.post(
    api_url,
    files={"file": img_bytes},
    params={"format": "image", "annotated": "true"}
)


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
    from roboflow import Roboflow
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
            

        except Exception as e:
            st.error(f"❌ Upload failed: {str(e)}")
    else:
        st.warning("⚠️ No image available to upload.")






