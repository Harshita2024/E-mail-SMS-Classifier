# E-mail-SMS-Classifier
## Introduction
The Email/SMS Spam Classifier is a machine learning-based application that identifies whether a given message is Spam or Not Spam. The project is built using Python and deployed with Streamlit for an interactive web interface. By leveraging Natural Language Processing (NLP) techniques and a trained classification model, this tool can be used to filter unwanted messages in real-time.<br>
## Concept
The project uses Text Preprocessing and TF-IDF (Term Frequency–Inverse Document Frequency) vectorization to convert textual messages into numerical features. These features are then fed into a pre-trained classification model (e.g., Naïve Bayes, Logistic Regression, etc.) to predict whether a message is spam.<br>
## The system works as follows:
1. User Input: The user enters a message (Email/SMS) into the Streamlit app.<br>
2. Preprocessing:
 - Lowercasing
 - Tokenization using NLTK
 - Removal of stopwords and punctuation
 - Stemming using Porter Stemmer <br>
3. Feature Extraction: Using a pre-trained TF-IDF Vectorizer. <br>
4. Prediction: The model predicts whether the message is spam (1) or not spam (0). <br>
5. Result Display: Output is shown in a clean, user-friendly interface. <br>
## Libraries Used
 - nltk – For tokenization, stopword removal, and stemming.
 - streamlit – For creating the interactive web interface.
 - pickle – For loading the saved model and vectorizer.
 - string – For handling punctuation removal.<br>
 ## Project Flow
  1. Data Preprocessing
 The text is transformed using the transform_text() function:
 - Convert text to lowercase.
 - Tokenize into words.
 - Remove non-alphanumeric tokens.
 - Remove stopwords & punctuation.
 - Apply stemming. <br>
 2. Model & Vectorizer Loading
 The pre-trained TF-IDF Vectorizer and classification model are loaded from vectorizer.pkl and model.pkl using pickle. <br>
 3. Prediction
 - The preprocessed text is transformed into a TF-IDF vector.
 - The model predicts the message category (Spam or Not Spam).
 - Prediction probabilities are also computed for transparency. <br>
 4. Deployment with Streamlit
 - The application is deployed using Streamlit, allowing users to interact with the model via a web interface. <br>
## How to Run Locally
 1. Clone the repository
    git clone <repo_link>
    cd spam-classifier<br>
 2. Install dependencies
    pip install -r requirements.txt<br>
 3. Download NLTK Data (only first time)
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')<br>
 4. Ensure model files exist
    Place vectorizer.pkl and model.pkl in the project directory.<br>
5.  Run the application
    streamlit run app.py<br>
## Conclusion
This Email/SMS Spam Classifier demonstrates the practical application of Natural Language Processing and Machine Learning in spam detection. With a simple and interactive web interface, it offers an efficient solution for real-time message filtering, reducing unwanted communications and improving user productivity. <br>
    

