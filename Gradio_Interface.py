import gradio as gr
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# === Setup model directory and available models ===
model_dir = "saved_models"
available_models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

# === Features from your dataset ===
feature_names = [
    "cap-diameter", "cap-shape", "cap-surface", "cap-color",
    "does-bruise-or-bleed", "gill-attachment", "gill-spacing", "gill-color",
    "stem-height", "stem-width", "stem-root", "stem-surface", "stem-color",
    "veil-color", "has-ring", "ring-type", "spore-print-color",
    "habitat", "season"
]
numerical_features = ["cap-diameter", "stem-height", "stem-width"]

choices = {
    "cap-shape": ['b', 'c', 'x', 'f', 's', 'p', 'o'],
    "cap-surface": ['i', 'g', 'y', 's', 'h', 'l', 'k', 't', 'w', 'e'],
    "cap-color": ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k'],
    "does-bruise-or-bleed": ['t', 'f'],
    "gill-attachment": ['a', 'x', 'd', 'e', 's', 'p', 'f', '?'],
    "gill-spacing": ['c', 'd', 'f'],
    "gill-color": ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k', 'f'],
    "stem-root": ['b', 's', 'c', 'u', 'e', 'z', 'r'],
    "stem-surface": ['i', 'g', 'y', 's', 'h', 'l', 'k', 't', 'w', 'e', 'f'],
    "stem-color": ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k', 'f'],
    "veil-color": ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k', 'f'],
    "has-ring": ['t', 'f'],
    "ring-type": ['c', 'e', 'r', 'g', 'l', 'p', 's', 'z', 'y', 'm', 'f', '?'],
    "spore-print-color": ['n', 'b', 'g', 'r', 'p', 'u', 'e', 'w', 'y', 'l', 'o', 'k'],
    "habitat": ['g', 'l', 'm', 'p', 'h', 'u', 'w', 'd'],
    "season": ['s', 'u', 'a', 'w'],
}

# === Predict function ===
def predict(model_file, *features):
    if not model_file:
        return "Please select a model ❗", None

    model_path = os.path.join(model_dir, model_file)
    model = joblib.load(model_path)
    input_df = pd.DataFrame([features], columns=feature_names)
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        return f"Prediction error ❌: {e}", None

    fig = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(input_df)[0]
            labels = model.classes_
            fig, ax = plt.subplots()
            ax.bar(labels, proba, color=['green' if lbl == 'e' else 'red' for lbl in labels])
            ax.set_title("Class Probability")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            plt.tight_layout()
        except Exception as e:
            print("⚠️ Plot error:", e)

    emoji = "✅" if prediction == "e" else "☠️"
    label = "Edible 🍽" if prediction == "e" else "Poisonous ☠️"
    return f"{emoji} Prediction: {label}", fig

# === UI layout ===
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🍄 Mushroom Edibility Classifier")
    gr.Markdown("Predict whether a mushroom is **edible** or **poisonous** using a trained model.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ➤ Input features")
            model_input = gr.Dropdown(choices=available_models, label="Select a model")
            feature_components = []
            for feat in feature_names:
                if feat in numerical_features:
                    feature_components.append(gr.Number(label=feat))
                else:
                    feature_components.append(gr.Dropdown(choices=choices[feat], label=feat))

        with gr.Column(scale=1):
            gr.Markdown("### ➤ Prediction")
            prediction_text = gr.Textbox(label="Prediction")
            prediction_plot = gr.Plot(label="Prediction Probability")

    predict_btn = gr.Button("Predict 🍄")
    predict_btn.click(
        fn=predict,
        inputs=[model_input] + feature_components,
        outputs=[prediction_text, prediction_plot]
    )

app.launch()
