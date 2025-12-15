# sentiment_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Setup
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# ========== 1. LOAD DATA ==========
train = pd.read_csv("twitter_training.csv", header=None, names=["entity", "sentiment", "empty", "text"])
val = pd.read_csv("twitter_validation.csv", header=None, names=["entity", "sentiment", "empty", "text"])

# Handle missing values
train['text'] = train['text'].fillna('')
val['text'] = val['text'].fillna('')

print("Training Shape:", train.shape)
print("Validation Shape:", val.shape)

print("\nTop Entities in Training Set:")
print(train['entity'].value_counts().head(10))

print("\nSentiment Distribution in Training Set:")
print(train['sentiment'].value_counts())

# ========== 2. EDA ==========
# Length of tweets
train['length'] = train['text'].apply(len)

sns.countplot(x='sentiment', data=train)
plt.title('Sentiment Distribution - Training Set')
plt.savefig("sentiment_distribution.png")
plt.clf()

sns.histplot(data=train, x='length', hue='sentiment', bins=30, element='step')
plt.title("Tweet Length Distribution by Sentiment")
plt.savefig("tweet_length_distribution.png")
plt.clf()

# ========== 3. TEXT CLEANING ==========
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text).lower()
    tokens = [word for word in text.split() if word not in stop]
    return ' '.join(tokens)

train['clean'] = train['text'].apply(clean_text)
val['clean'] = val['text'].apply(clean_text)

# ========== 4. WORD CLOUDS ==========
for sentiment in ['Positive', 'Negative', 'Neutral']:
    subset = train[train['sentiment'] == sentiment]
    text = ' '.join(subset['clean'])
    if text:
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'WordCloud - {sentiment}')
        plt.savefig(f'wordcloud_{sentiment.lower()}.png')
        plt.clf()

# ========== 5. TF-IDF + MODEL TRAINING ==========
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(train['clean'])
X_val = tfidf.transform(val['clean'])
y_train = train['sentiment']
y_val = val['sentiment']

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# ========== 6. CONFUSION MATRIX ==========
cm = confusion_matrix(y_val, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.clf()

# ========== 7. ENTITY-WISE SENTIMENT ==========
plt.figure(figsize=(10, 6))
top_entities = train['entity'].value_counts().head(10).index.tolist()
subset = train[train['entity'].isin(top_entities)]
sns.countplot(y='entity', hue='sentiment', data=subset, order=top_entities)
plt.title('Top 10 Entities Sentiment Breakdown')
plt.tight_layout()
plt.savefig("entity_sentiment_breakdown.png")
plt.clf()

print("\n✅ All steps completed successfully. Check PNGs for visualizations.")
