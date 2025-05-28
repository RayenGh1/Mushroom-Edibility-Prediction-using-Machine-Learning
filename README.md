# 🍄 Mushroom Classification Project (AI/ML)

This project applies several machine learning algorithms to classify mushrooms using a custom dataset.  
It includes steps for data exploration, feature engineering, model training, evaluation, and fine-tuning.

---

## 📁 Project Structure

Project_MushroomV2/
│
├── Dataset/ # Raw + split CSVs (excluded from GitHub)
├── plots/ # Confusion matrix images
├── saved_models/ # Trained models (.pkl files)
│
├── 1_DataStructure.py # Data loading, encoding, correlation, visualizations
├── 2_SplittingData.py # Stratified split on stem-width
├── 3_Preprocessing.py # Model training, evaluation, hyperparameter tuning
├── Gradio_Interface.py # (Optional) GUI or demo interface
├── Documentation.mushroom.pdf # Report (excluded from GitHub)
└── README.md

---

## 🧠 Models Used

- Decision Tree (base & tuned)
- Random Forest (base & tuned)
- k-Nearest Neighbors (base & tuned)
- Support Vector Machine (base)
- Naive Bayes (base)
- SGD Classifier (base)

---

## 🧪 How to Run

1. Place `secondary_data.csv` inside the `Dataset/` folder.
2. Run the following scripts in order:
   - `1_DataStructure.py`
   - `2_SplittingData.py`
   - `3_Preprocessing.py`
3. Trained models will appear in `saved_models/`, and confusion matrices in `plots/`.

---

## 📚 Source of Inspiration

Some parts of the code and methodology are inspired by:

**Géron, A.** (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O’Reilly Media. ISBN: 978-1-098-12597-4
