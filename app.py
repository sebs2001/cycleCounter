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

if uploaded_file:
    # Convert to base64
    img_bytes = uploaded_file.getvalue()
    img_str = base64.b64encode(img_bytes).decode("utf-8")

    # ✅ INSERT your actual Roboflow API URL here (replace placeholders)
    api_url = "o9tbMpy3YklEF3MoRmdR"

    # Send image to Roboflow model
    response = requests.post(api_url, json={"image": img_str})
    result = response.json()

    # Show the image
    st.image(uploaded_file, caption="Original Image")

    # Show annotated image from Roboflow (if available)
    if "image" in result and "url" in result["image"]:
        st.image(result["image"]["url"], caption="Roboflow Detection")

    # Show count of detected objects
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
