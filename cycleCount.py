import streamlit as st
import requests
from roboflow import Roboflow

st.title("Electronic Components Detection App")

# Capture photo using Streamlit's camera input
img = st.camera_input("Take a picture of your electronic components")

if img is not None:
    # Display the original image
    st.image(img, caption="Original Image", use_column_width=True)

    # Send the image to Roboflow's object detection model
    # Roboflow inference API endpoint format:
    # https://detect.roboflow.com/<MODEL_NAME>/<VERSION>?api_key=<API_KEY>
    url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"
    files = {"file": ("image.jpg", img.getvalue(), "image/jpeg")}
    response = requests.post(url, files=files)
    result = response.json()

    # Display the annotated image with bounding boxes
    annotated_url = result["image"]["url"]
    st.image(annotated_url, caption="Annotated Image", use_column_width=True)

    # Show the count of detected objects
    count = len(result["predictions"])
    st.write("Detected objects:", count)

    # Button to upload the image for future annotation
    if st.button("Upload for Annotation"):
        # Save the image temporarily for upload
        with open("upload.jpg", "wb") as f:
            f.write(img.getvalue())
        
        # Use Roboflow SDK to upload the image
        rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
        project = rf.workspace("quanticwork").project("my-first-project-eintr")
        project.upload(
            image_path="upload.jpg",
            batch_name="streamlit-submits",
            split="train",
            num_retry_uploads=3,
            tag_names=["from-streamlit"],
            sequence_number=99,
            sequence_size=100
        )
        st.write("Image uploaded for future annotation!")





