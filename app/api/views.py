from django.shortcuts import render, redirect
from django.http import JsonResponse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

with open('ml_script/logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('ml_script/vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

stop_words = set(stopwords.words('english'))

def remove_stopwords_except_not(word_list):
    return [word for word in word_list if word.lower() == 'not' or word.lower() not in stop_words]

def clean_text(text):
    # Remove special characters and numbers, and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = remove_stopwords_except_not(tokens)
    # Join tokens back into a clean text
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def predict_sentiment(cleaned_text, rating):
    combined_input = f'{cleaned_text} {rating}'  

    # Vectorize the combined input using the loaded TF-IDF vectorizer
    combined_input_tfidf = tfidf_vectorizer.transform([combined_input])

    # Use the loaded sentiment analysis model to predict sentiment based on both text and rating
    sentiment = model.predict(combined_input_tfidf)

    return sentiment[0]
        

def predict_sentiment_view(request):
    if request.method == 'POST':
        review_text = request.POST.get('review_text')
        rating = int(request.POST.get('rating'))
        
        # Clean the review_text
        cleaned_review_text = clean_text(review_text)
        
        # Perform sentiment prediction using your model (replace with your actual model)
        sentiment = predict_sentiment(cleaned_review_text, rating)
        
        # Define a function to predict sentiment (replace with your actual model)
        # Return the predicted sentiment as JSON response
        return JsonResponse({'sentiment': sentiment})

    return render(request, 'home.html')