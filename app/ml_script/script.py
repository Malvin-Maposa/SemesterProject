from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import  string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


df = pd.read_csv('sneakers_Reviews_Dataset.csv', sep = ';')

df['review_text'] = df['review_text'].str.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

df['review_text'] = df['review_text'].apply(remove_punctuation)

df['tokenized_text'] = df['review_text'].apply(word_tokenize)

stop_words = set(stopwords.words('english'))
def remove_stopwords_except_not(word_list):
    return [word for word in word_list if word.lower() == 'not' or word.lower() not in stop_words]
df['tokenized_text'] = df['tokenized_text'].apply(remove_stopwords_except_not)
df['tokenized_text'] = df['tokenized_text'].apply(lambda x: ' '.join(x))

df['tokenized_text'] = df['tokenized_text'].apply(lambda tokens: ''.join(tokens))

df = df.drop(columns=['review_text'])

df = df.rename(columns={'tokenized_text': 'review_text'})

sentiment_mapping = {
    5: 'Positive',
    4: 'Positive',
    3: 'Neutral',
    2: 'Negative',
    1: 'Negative'
}

df['sentiment'] = df['rating'].map(sentiment_mapping)

X = df[['review_text', 'rating']].astype(str).agg(' '.join, axis=1)
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

logistic_regression_model = LogisticRegression()

logistic_regression_model.fit(X_train_tfidf, y_train)

with open('logistic_model.pkl', 'wb') as model_file:
    pickle.dump(logistic_regression_model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

with open('logistic_model.pkl', 'rb') as model_file:
    logistic_regression_model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

user_input = ["it is great", 3]

for i in range(len(user_input)):
    # Tokenize text
    tokens = word_tokenize(user_input[i])
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    user_input[i] = ' '.join(tokens)

user_input_vectorized = tfidf_vectorizer.transform(user_input)

predicted_sentiment = logistic_regression_model.predict(user_input_vectorized)

print(f"User sentiment: {predicted_sentiment[0]}")