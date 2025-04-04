import os
import requests
import streamlit as st
from roboflow import Roboflow
from collections import Counter

# 1) CREATE A FOLDER (IF NEEDED)
os.makedirs("saved", exist_ok=True)

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

# 2) CAMERA INPUT
uploaded_file = st.camera_input("Take a photo", key="camera1")

if uploaded_file:
    img_bytes = uploaded_file.getvalue()
    # Save the original image locally so we can upload it later
    original_path = "temp_upload.jpg"
    with open(original_path, "wb") as f:
        f.write(img_bytes)

    # 3) INFERENCE ENDPOINT (UPDATE YOUR VERSION & API KEY AS NEEDED)
    api_url = "https://detect.roboflow.com/my-first-project-eintr/9"
    api_key = "o9tbMpy3YklEF3MoRmdR"

    # 3A) GET & DISPLAY ANNOTATED IMAGE FOR VISUAL REFERENCE
    response_image = requests.post(
        f"{api_url}?api_key={api_key}",
        files={"file": img_bytes},
        params={"format": "image", "annotated": "true"}
    )
    if response_image.status_code == 200:
        st.image(response_image.content, caption="Annotated Image with bounding boxes")
    else:
        st.error(f"Error fetching annotated image: {response_image.status_code}")

    # 3B) GET JSON PREDICTIONS (ACTUAL COORDINATES)
    response_json = requests.post(
        f"{api_url}?api_key={api_key}",
        files={"file": img_bytes},
        params={"format": "json"}
    )
    if response_json.status_code == 200:
        data = response_json.json()
        predictions = data.get("predictions", [])
        if predictions:
            # Convert prediction classes to uppercase to standardize, then count each class
            counts = Counter(p.get("class", "").upper() for p in predictions)
            resistor_count = counts.get("RESISTOR", 0)
            check_count = counts.get("CHECK", 0)
            # Display a message showing both classes
            st.success(f"Detected:\n- {resistor_count} resistor(s)\n- {check_count} CHECK object(s)")
            # Optional: Debug output of the complete counts (remove if not needed)
            st.write("Detailed counts:", dict(counts))
            # Store predictions in session state so we can upload them as annotations
            st.session_state["predictions"] = predictions
        else:
            st.warning("No predictions returned.")
    else:
        st.error(f"Error fetching predictions: {response_json.status_code}")

# 4) UPLOAD THE IMAGE + ANNOTATIONS TO ROBOFLOW
#    This time we pass bounding-box data so Roboflow sees it as 'annotated.'
if st.button("💾 Upload to Roboflow"):
    if not uploaded_file:
        st.warning("⚠️ No image available to upload.")
    else:
        if "predictions" not in st.session_state:
            st.warning("⚠️ No bounding-box data to upload. (Run inference first.)")
        else:
            # Convert predictions to Roboflow's annotation format
            annotations = []
            for p in st.session_state["predictions"]:
                annotations.append({
                    "x": p["x"],
                    "y": p["y"],
                    "width": p["width"],
                    "height": p["height"],
                    "class": p["class"]
                })

            try:
                # Initialize Roboflow
                rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
                project = rf.workspace("quanticwork").project("my-first-project-eintr")

                # Upload original image WITH annotation data
                upload_response = project.upload(
                    image_path="temp_upload.jpg",
                    annotation={"annotations": annotations}
                )

                st.success("✅ Auto-labeled image uploaded to Roboflow! Check 'Annotate' or 'Dataset' in your project.")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")






