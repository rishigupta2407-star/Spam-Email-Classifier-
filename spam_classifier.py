# Spam Email Classifier

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset (you can replace with CSV later)
data = {
    'email': [
        'Win money now!!!',
        'Hello friend how are you',
        'Claim your free prize',
        'Meeting at 5 pm',
        'Congratulations you won lottery',
        'Let’s catch up tomorrow'
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham
}

df = pd.DataFrame(data)

# Features and labels
X = df['email']
y = df['label']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert text to numerical
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom input
test_email = ["Free gift waiting for you"]
test_vec = vectorizer.transform(test_email)
prediction = model.predict(test_vec)

print("Spam" if prediction[0] == 1 else "Not Spam")