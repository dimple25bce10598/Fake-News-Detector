# Fake News Detector (No external dataset needed)

import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# ----------------------------
# 1. Sample Dataset (BUILT-IN)
# ----------------------------
data = pd.DataFrame({
    'text': [
        "Breaking: Government launches new education policy",
        "Scientists discover new species in ocean",
        "You won a lottery of 5 crore click this link",
        "Earn money fast with this secret trick",
        "PM announces new infrastructure projects",
        "Fake news spreading about celebrity death",
        "Click here to claim your prize now",
        "Study shows benefits of daily exercise"
    ],
    'label': [
        "REAL", "REAL", "FAKE", "FAKE",
        "REAL", "FAKE", "FAKE", "REAL"
    ]
})

# ----------------------------
# 2. Clean Text
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

data['text'] = data['text'].apply(clean_text)

# ----------------------------
# 3. Convert Labels
# ----------------------------
data['label'] = data['label'].map({'FAKE': 0, 'REAL': 1})

# ----------------------------
# 4. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.25, random_state=42
)

# ----------------------------
# 5. Vectorization
# ----------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 6. Train Model
# ----------------------------
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# ----------------------------
# 7. Prediction Function
# ----------------------------
def predict_news(news):
    news = clean_text(news)
    vec = vectorizer.transform([news])
    pred = model.predict(vec)[0]
    return "REAL" if pred == 1 else "FAKE"

# ----------------------------
# 8. Test
# ----------------------------
print("Model is ready!\n")

while True:
    user_input = input("Enter news (or 'exit'): ")
    if user_input.lower() == 'exit':
        break
    print("Prediction:", predict_news(user_input))
