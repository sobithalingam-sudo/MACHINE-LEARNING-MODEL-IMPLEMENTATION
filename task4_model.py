import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'text': [
        'Free entry in 2 a weekly competition!',
        'Hey, are we meeting today?',
        'Congratulations! You have won a prize.',
        'Please call me back.',
        'Win a $1000 gift card now!',
        'Can we have lunch tomorrow?'
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example prediction
sample_text = ["Win a free ticket now!"]
sample_vector = vectorizer.transform(sample_text)
prediction = model.predict(sample_vector)
print("\nSample Prediction:", "Spam" if prediction[0]==1 else "Not Spam")
