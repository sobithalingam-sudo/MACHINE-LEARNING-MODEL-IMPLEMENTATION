import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Sample dataset ----------------
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

# ---------------- Train the model ----------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ---------------- Visualization ----------------
sns.set_style("whitegrid")

# 1. Spam vs Non-Spam count using Matplotlib (no warnings)
counts = df['label'].value_counts()  # 0 = Not Spam, 1 = Spam
labels = ['Not Spam', 'Spam']

plt.figure(figsize=(6,4))
plt.bar(labels, counts, color=['#440154', '#21908C'])
plt.title("Spam vs Non-Spam Count")
plt.ylabel("Number of Messages")
plt.xlabel("Message Type")
plt.show()

# 2. Predictions for sample messages
sample_texts = [
    "Win a free ticket now!",
    "Let's meet tomorrow for lunch",
    "Congratulations! You won a prize!"
]

sample_vectors = vectorizer.transform(sample_texts)
predictions = model.predict(sample_vectors)

plt.figure(figsize=(8,4))
sns.barplot(x=sample_texts, y=predictions, palette='coolwarm')
plt.ylim(0,1.5)
plt.ylabel("Prediction (0=Not Spam, 1=Spam)")
plt.title("Predictions for Sample Messages")
plt.xticks(rotation=25, ha='right')
plt.show()
