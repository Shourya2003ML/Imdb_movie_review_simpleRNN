# IMDb Movie Review Sentiment Analysis using Simple RNN 🎬🧠

An end-to-end Natural Language Processing (NLP) project that classifies IMDb movie reviews as either **Positive** or **Negative**. This project implements a Simple Recurrent Neural Network (RNN) to understand sequential text data and is deployed through an interactive Streamlit web application.

## 🚀 Project Overview

Sentiment analysis is a fundamental task in NLP. This project utilizes the widely recognized IMDb dataset of 50,000 movie reviews. A Deep Learning model built with TensorFlow/Keras learns the contextual flow of words using a Simple RNN architecture, predicting the underlying sentiment of unseen movie reviews in real-time via a user-friendly UI.

## 💡 Key Learnings & Takeaways

Building this project provided deep hands-on experience with several core NLP and Deep Learning concepts:
* **Sequential Data Processing:** Learned how Recurrent Neural Networks (RNNs) maintain a "memory" of previous inputs using hidden states, making them well-suited for text where word order matters.
* **Text Preprocessing:** Mastered the pipeline of converting raw text into machine-readable formats, including tokenization, vocabulary indexing, and sequence padding (ensuring uniform input shapes).
* **Word Embeddings:** Understood how embedding layers map sparse, high-dimensional word indices into dense, lower-dimensional semantic vectors.
* **Model Limitations:** Gained practical insight into the "vanishing gradient problem" inherent in Simple RNNs when processing long reviews, establishing a foundation for advanced architectures like LSTMs and GRUs.
* **End-to-End Deployment:** Successfully bridged the gap between a Jupyter Notebook experiment and a live web application using Streamlit, managing model serialization and inference pipelines.

## 🛠️ Tech Stack

* **Deep Learning Framework:** TensorFlow / Keras
* **Web Framework:** Streamlit
* **Data Manipulation & NLP:** NumPy, Pandas, Keras Preprocessing
* **Environment:** Python 3.x, Jupyter Notebook

## 📁 Repository Structure

```text
Imdb_movie_review_simpleRNN/
├── 🌐 app.py                       # Streamlit application for the web UI
├── 📓 experiments.ipynb            # Data loading, text preprocessing, and RNN training
├── 🧠 simple_rnn_model.h5          # Saved trained Simple RNN model
├── 📚 word_index.json              # Saved dictionary mapping words to integer indices
├── 📦 requirements.txt             # Required Python dependencies
├── 📜 LICENSE                      # Project License
└── 📖 README.md                    # Project documentation
```
*(Note: Ensure the file names match your exact repository structure, e.g., `experiments.ipynb` or `model.h5`)*

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Shourya2003ML/Imdb_movie_review_simpleRNN.git](https://github.com/Shourya2003ML/Imdb_movie_review_simpleRNN.git)
   cd Imdb_movie_review_simpleRNN
   ```

2. **Create a virtual environment (Optional but highly recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

To launch the interactive web application, run the following command in your terminal:

```bash
streamlit run app.py
```

1. Open the provided `localhost` URL in your web browser.
2. Type or paste any movie review into the text area.
3. Click **Predict** to see if the Simple RNN classifies the sentiment as Positive or Negative, along with the confidence score!

## 🧠 Architecture Details

1. **Embedding Layer:** Converts integer-encoded vocabulary words into dense vectors of fixed size.
2. **SimpleRNN Layer:** Processes the embedded sequences, capturing sequential relationships and context.
3. **Dense Output Layer:** A fully connected layer with a Sigmoid activation function to output a probability between 0 (Negative) and 1 (Positive).
