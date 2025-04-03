import streamlit as st
import requests
import os

# Ensure a folder exists for saving images if needed
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# Camera input widget (mobile and desktop compatible)
uploaded_file = st.camera_input("Take a photo", key="camera1")

if uploaded_file is not None:
    # Convert the uploaded image to bytes
    img_bytes = uploaded_file.getvalue()

    # Roboflow inference API endpoint (update with your project version if needed)
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    ### PART 1: Inference & Annotated Image Retrieval ###

    # Request JSON with annotated image URL (bounding boxes included)
    response_json = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "json", "annotated": "true"}
    )

    if response_json.status_code != 200:
        st.error(f"API Error (JSON): {response_json.status_code}")
        st.stop()

    # Parse the JSON response
    result = response_json.json()

    # Display the raw JSON for debugging
    st.subheader("Raw JSON Predictions")
    st.json(result)

    # Display the original image
    st.subheader("Original Image")
    st.image(uploaded_file, caption="Original Image")

    # Attempt to display the annotated image using the URL in the JSON response
    if "image" in result and "url" in result["image"]:
        st.subheader("Annotated Image (from JSON URL)")
        st.image(result["image"]["url"], caption="Annotated Image with bounding boxes (via JSON)")
    else:
        st.warning("No annotated image URL found in the JSON response.")

    # Additionally, bypass JSON and request the annotated image directly
    response_annotated = requests.post(
        api_url,
        files={"file": img_bytes},
        params={"format": "image", "annotated": "true"}
    )

    if response_annotated.status_code == 200:
        st.subheader("Annotated Image (direct binary response)")
        st.image(response_annotated.content, caption="Annotated Image with bounding boxes (direct)")
    else:
        st.error(f"API Error (direct annotated image): {response_annotated.status_code}")

    # Count the number of predictions (e.g., resistors)
    if "predictions" in result:
        count = len(result["predictions"])
        st.success(f"Detected {count} resistors.")
    else:
        st.warning("No predictions returned.")

### PART 2: Direct Upload to Roboflow via Requests ###
if st.button("💾 Upload to Roboflow"):
    if uploaded_file:
        # Save the uploaded image temporarily
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Define your dataset name (workspace/project) and API key
        dataset_name = "quanticwork/my-first-project-eintr"  # update if necessary
        api_key = "o9tbMpy3YklEF3MoRmdR"
        upload_url = f"https://api.roboflow.com/dataset/{dataset_name}/upload"

        # Define upload parameters (adjust split, batch, and tag as needed)
        params = {
            "api_key": api_key,
            "name": "temp_upload.jpg",
            "split": "train",
            "batch": "streamlit-submits",
            "tag": "from-streamlit"
        }

        # Upload the image using a direct POST request
        with open(temp_path, "rb") as f:
            files = {"file": f}
            upload_response = requests.post(upload_url, files=files, params=params)

        if upload_response.status_code == 200:
            st.success("✅ Uploaded to Roboflow. Check Annotate > Unannotated.")
            st.subheader("Upload Response")
            st.json(upload_response.json())
        else:
            st.error(f"❌ Upload failed: {upload_response.status_code}\n{upload_response.text}")
    else:
        st.warning("⚠️ No image available to upload.")
