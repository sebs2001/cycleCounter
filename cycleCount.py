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
uploaded_file = st.camera_input("Take a photo", key="camera1")


if uploaded_file:
    # Convert to base64
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    # ✅ Correct Roboflow API URL format
    api_url = "https://detect.roboflow.com/my-first-project-8/1?api_key=o9tbMpy3YklEF3MoRmdR"



if uploaded_file is not None:
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    api_url = "https://detect.roboflow.com/my-first-project-8/1?api_key=o9tbMpy3YklEF3MoRmdR"
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
    if st.button("💾 Save annotated image"):
        if "image" in result and "url" in result["image"]:
            image_data = requests.get(result["image"]["url"]).content
            image_id = len(os.listdir("saved"))
            with open(f"saved/resistor_{image_id}.jpg", "wb") as f:
                f.write(image_data)
            st.success("Annotated image saved.")
