import pandas as pd
import re
import emoji
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

nltk.download('stopwords')

# Membaca dataset dari file CSV
df = pd.read_csv('dataset_algoritma.csv')

# Preprocessing Class
class TextPreprocessor:
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('indonesian'))

    def cleansing(self, text):
        """
        Membersihkan teks dari karakter khusus, emoji, URL, dan stopwords.
        """
        text = re.sub(r'\B@\w+', '', text)  # Remove mentions
        text = emoji.demojize(text)  # Replace emoji
        text = re.sub(r'(http|https):\/\/\S+', '', text)  # Remove URLs
        text = re.sub(r'#+', '', text)  # Remove hashtags
        text = text.lower()  # Lowercase
        text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic chars
        text = contractions.fix(text)  # Expand contractions
        words = text.split()
        words = [word for word in words if word not in self.stop_words]  # Remove stopwords
        words = [self.stemmer.stem(word) for word in words]  # Stemming
        return ' '.join(words)

# Preprocess data
preprocessor = TextPreprocessor()
df["Cleaned_Input"] = df["Input"].apply(preprocessor.cleansing)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Input"])

def chatbot_recommendation(user_input):
    """
    Memberikan rekomendasi algoritma berdasarkan input pengguna.
    Jika input terlalu umum atau pendek, chatbot meminta klarifikasi.
    """
    # Periksa panjang input (menghindari input terlalu pendek)
    if len(user_input.split()) < 3:
        return {
            "Pertanyaan": user_input,
            "Jawaban": "Input Anda terlalu umum. Bisa jelaskan lebih detail tujuan dari pengolahan data yang Anda inginkan?",
            "Similarity Score": None
        }

    # Bersihkan input dan hitung skor kemiripan
    cleaned_input = preprocessor.cleansing(user_input)
    user_tfidf = vectorizer.transform([cleaned_input])
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    max_similarity = similarities.max()

    # Ambang batas minimum untuk rekomendasi
    recommendation_threshold = 0.1  # Skor cukup tinggi untuk rekomendasi

    if max_similarity > recommendation_threshold:
        max_index = similarities.argmax()
        return {
            "Pertanyaan": df["Input"][max_index],
            "Jawaban": df["Response"][max_index],
            "Similarity Score": max_similarity
        }
    else:
        return {
            "Pertanyaan": user_input,
            "Jawaban": "Saya kurang memahami maksud Anda. Bisa jelaskan lebih detail tujuan pengolahan data Anda?",
            "Similarity Score": max_similarity
        }


def classify_question(user_input):
    """
    Mengklasifikasikan apakah pertanyaan pengguna terkait machine learning.
    """
    ml_keywords = [
        "algoritma", "machine learning", "data", "klasifikasi", "regresi",
        "clustering", "model", "prediksi"
    ]

    if any(keyword in user_input.lower() for keyword in ml_keywords):
        return "ml_related", "Kami siap membantu! Pertanyaan Anda relevan dengan algoritma machine learning."

    return "not_related", "Pertanyaan Anda tidak terkait dengan algoritma machine learning yang tersedia."
