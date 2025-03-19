import joblib
import neattext.functions as nfx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split# Load dataset (Upsdate the path as per your system)
df = pd.read_csv("C:/Users/chunc/OneDrive/Desktop/mlproject/emotion_dataset_raw.csv")

df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)  # Remove @username
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)  # Remove stopwords

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Clean_Text'])
y = df['Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

joblib.dump(lr_model, "lr_emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
