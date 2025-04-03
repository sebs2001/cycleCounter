import streamlit as st
import requests
import os

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

    # Define your Roboflow inference endpoint (update project/version/API key as needed)
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    # --------------------------
    # Inference: Get the annotated image directly
    # --------------------------
    response_image = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "image", "annotated": "true"}
    )

    if response_image.status_code == 200:
        st.image(response_image.content, caption="Annotated Image with bounding boxes")
    else:
        st.error(f"Error fetching annotated image: {response_image.status_code}")

    # --------------------------
    # Inference: Get the prediction count via JSON (do not display raw JSON)
    # --------------------------
    response_json = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "json", "annotated": "true"}
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
# Upload Section: Direct upload to Roboflow using the REST API
# --------------------------
if st.button("💾 Upload to Roboflow"):
    if not uploaded_file:
        st.warning("No image to upload.")
    else:
        # Save the uploaded image temporarily
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Set your dataset info and API key (update if necessary)
        dataset_name = "quanticwork/my-first-project-eintr"  # Format: workspace/project-slug
        api_key = "o9tbMpy3YklEF3MoRmdR"
        upload_url = f"https://api.roboflow.com/dataset/{dataset_name}/upload"
        params = {
            "api_key": api_key,
            "name": "temp_upload.jpg",
            "split": "train",
            "batch": "streamlit-submits",
            "tag": "from-streamlit"
        }

        try:
            with open(temp_path, "rb") as f:
                files = {"file": f}
                upload_response = requests.post(upload_url, files=files, params=params)
            if upload_response.status_code == 200:
                st.success("✅ Uploaded to Roboflow. Check 'Annotate > Unannotated' in your project.")
            else:
                st.error(f"Upload failed: {upload_response.status_code}\n{upload_response.text}")
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")

