---

# 📩 SMS Spam Classifier

A Machine Learning-powered web app that classifies SMS messages as **Spam** or **Not Spam** in real-time using NLP and a trained model. Built with Python, scikit-learn, and Streamlit.


[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-brightgreen?style=for-the-badge\&logo=streamlit)](https://sms-spam-classifier-mlproject-cwrwdzkrmqpdy3ybxf5cib.streamlit.app/)

---

## 🚀 Live Demo

👉 **Try the app here**:
🔗 (https://sms-spam-classifier-mlproject-cwrwdzkrmqpdy3ybxf5cib.streamlit.app/)

---

## ✨ Features

* 🔍 Classifies any input SMS as **Spam** or **Not Spam**
* 🧠 Uses Natural Language Processing (NLP) for text preprocessing
* 📊 Displays prediction confidence score
* 📚 Jupyter Notebook included to explain training and evaluation
* 🗃️ Maintains a local prediction history (with option to delete)
* 🌐 Fully deployed and accessible online

---

## 🧠 How It Works

1. **Preprocessing:**

   * Lowercasing
   * Removing stopwords
   * Stemming using NLTK

2. **Vectorization:**

   * TF-IDF Vectorizer (`vectorizer.pkl`)

3. **Model:**

   * Trained using Naive Bayes / any scikit-learn classifier
   * Pickled as `model.pkl`

4. **Notebook:**

   * Full training pipeline and evaluation included in `spam_classifier_training.ipynb`

5. **UI:**

   * Built using Streamlit with custom CSS for a clean look
   * Automatically hides result when input is cleared
   * Tracks last 5 predictions locally

---

## 📦 Installation (For Local Run)

1. Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/sms-spam-classifier.git
cd sms-spam-classifier
```

2. (Optional) Create a virtual environment:

```
python -m venv .venv
.\.venv\Scripts\activate  # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the Streamlit app:

```
streamlit run app.py
```

5. (Optional) Run the notebook:

You can open `spam_classifier_training.ipynb` in **Jupyter Notebook** or **VS Code** to explore how the model was trained.

---

## 📁 Project Structure

```
sms-spam-classifier/
│
├── app.py                        # Streamlit frontend logic
├── model.pkl                     # Trained ML model
├── vectorizer.pkl                # TF-IDF vectorizer
├── requirements.txt              # Project dependencies
├── spam_classifier_training.ipynb # Jupyter Notebook with model training code
└── README.md                     # Documentation (this file)
```

---

## 🛠️ Technologies Used

* Python 
* scikit-learn 
* NLTK 
* Streamlit 
* Jupyter Notebook 
* Pandas, Matplotlib, Seaborn (optional for EDA)

---

## 👤 Author

Made by Aaryan Lunis

---

## 📃 License

This project is open-source and available under the MIT License.

---
