import streamlit as st
import requests
import base64
import os

# Make sure save folder exists
if not os.path.exists("saved"):
    os.makedirs("saved")

st.title("Cycle Counter v1.0")
st.write("Take a photo.")

uploaded_file = st.camera_input("Take a photo")

if uploaded_file:
    # Convert to base64
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    # Your actual Roboflow model endpoint
    api_url = "https://detect.roboflow.com/my-first-project-eintr/8?api_key=o9tbMpy3YklEF3MoRmdR"

    response = requests.post(api_url, json={"image": img_str})
    result = response.json()  # ← this will work if Roboflow responds correctly

    # Show original image
    st.image(uploaded_file, caption="Original Image")

    # ✅ JUST THIS LINE: show the annotated bounding box result
    if "image" in result and "url" in result["image"]:
        st.image(result["image"]["url"], caption="Detection with bounding boxes")

    # Count results
    if "predictions" in result:
        count = len(result["predictions"])
        st.success(f"Detected {count} resistors.")
    else:
        st.warning("No predictions returned.")

    # Upload button (optional)
    if st.button("💾 Upload to Roboflow"):
        from roboflow import Roboflow
        rf = Roboflow(api_key="o9tbMpy3YklEF3MoRmdR")
        project = rf.workspace("quanticwork").project("my-first-project-eintr")

        # Save original image
        with open("temp.jpg", "wb") as f:
            f.write(img_bytes)

        project.upload("temp.jpg", batch_name="streamlit-submits", split="train")
        st.success("✅ Uploaded to Roboflow.")




