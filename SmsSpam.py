import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# Train the model
model = LogisticRegression()
model.fit(X_train_features, y_train)

# Evaluate the model on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print("Accuracy on training data:", accuracy_on_training_data)

# Evaluate the model on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print("Accuracy on test data:", accuracy_on_test_data)

# Print the classification report
print(classification_report(y_test, prediction_on_test_data))

# Predict on new SMS input
input_sms = input("Enter the sms:")

input_data_features = feature_extraction.transform([input_sms])
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("The SMS is not Spam")
else:
    print("Spam SMS")
