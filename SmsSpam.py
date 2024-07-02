import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB  # Add Naive Bayes
from sklearn.svm import LinearSVC  # Add SVM
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv(r"C:\Users\AKSHAT SOMANI\OneDrive\Desktop\home\gla akshat somani\codesoft\SmsSpam\mail_data.csv")

# Replace null values with empty strings
df = df.where((pd.notnull(df)), "")

# Encode the Category column: spam=0, ham=1
df.loc[df['Category'] == 'spam', 'Category'] = 0
df.loc[df['Category'] == 'ham', 'Category'] = 1

# Features and Labels
X = df['Message']
y = df['Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Feature extraction using TF-IDF Vectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert y_train and y_test to integers
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Train the models: Logistic Regression, Naive Bayes, SVM
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

best_model = None
best_accuracy = 0.0

for name, model in models.items():
    model.fit(X_train_features, y_train)
    
    # Evaluate the model on test data
    predictions = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n{name}")
    print(f"Accuracy on test data: {accuracy}")
    print(classification_report(y_test, predictions))
    
    # Track the best model based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Predict on new SMS input using the best model
input_sms = input("Enter the sms: ")
input_data_features = feature_extraction.transform([input_sms])
prediction = best_model.predict(input_data_features)

if prediction[0] == 1:
    print("Not Spam")
else:
    print("Spam")
