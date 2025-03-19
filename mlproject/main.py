import joblib

lr_model = joblib.load("lr_emotion_model.pkl")
svm_model = joblib.load("svm_emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_emotion(text):
    
    text_vectorized = vectorizer.transform([text])
    
  
    lr_prediction = lr_model.predict(text_vectorized)[0]
    svm_prediction = svm_model.predict(text_vectorized)[0]
    
   
    print(f"Logistic Regression Prediction: {lr_prediction}")
    print(f"SVM Prediction: {svm_prediction}")
    
    
    return lr_prediction, svm_prediction


if __name__ == "__main__":
    user_text = input("Enter a sentence to analyze emotion: ")
    predict_emotion(user_text)
