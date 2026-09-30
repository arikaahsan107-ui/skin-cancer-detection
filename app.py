import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Cancer Detection", page_icon="🩺", layout="centered")

@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("model.keras", compile=False)
    cfg = json.load(open("config.json"))
    names = json.load(open("class_names.json"))
    return model, cfg, names

model, cfg, names = load_all()
S = cfg["img_size"]

st.title("🩺 Skin Lesion Classifier")
st.caption(f"Model: {cfg['model']} | Test accuracy: {cfg['test_accuracy']*100:.1f}% | Macro-F1: {cfg['test_f1_macro']*100:.1f}%")

file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    x = tf.image.resize(np.array(img), (S, S)).numpy()[None].astype("float32")
    probs = model.predict(x, verbose=0)[0]
    top = int(np.argmax(probs))
    st.subheader(f"Prediction: {names[top]}  ({probs[top]*100:.1f}%)")
    st.bar_chart({n: float(p) for n, p in zip(names, probs)})

st.warning("Educational/research tool only. Not a medical diagnosis. Please consult a dermatologist.")
