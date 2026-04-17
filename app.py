import torch
import torch.nn.functional as F
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2

from model_architecture import CNN

@st.cache_resource
def load_model():
    model = CNN()
    checkpoint = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model = load_model()


st.title("MNIST Digit Recognizer")
st.write("Draw a digit below:")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------------
# Prediction
# ---------------------------
if canvas_result.image_data is not None:


    img = canvas_result.image_data[:, :, :3]
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)


    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)


    img = img / 255.0

    if img.mean() > 0.5:
        img = 1 - img


    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # ---------------------------
    st.image(img, caption="Model Input (28×28)", width=150, clamp=True)


    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]

    pred = probs.argmax().item()
    confidence = probs[pred].item()

    st.subheader(f"Prediction: {pred}")
    st.write(f"Confidence: {confidence:.4f}")

    st.bar_chart({str(i): float(probs[i]) for i in range(10)})