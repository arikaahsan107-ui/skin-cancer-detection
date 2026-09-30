# Streamlit Cloud deployment
1. Create a GitHub repo and upload the files in this folder (app.py, model.keras, class_names.json, config.json, requirements.txt).
2. share.streamlit.io -> New app -> select repo -> main file: app.py
3. In Advanced settings choose Python 3.11 or 3.12 (must be supported by the TensorFlow version in requirements.txt).
4. If model.keras is > 100 MB, use Git LFS (git lfs track "*.keras") or pick a lighter model (MobileNetV2 / EfficientNetB0).
