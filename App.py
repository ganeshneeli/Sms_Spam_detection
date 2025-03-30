from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import re

# Initialize the Flask app
app = Flask(__name__)

# Define paths
input_file_path_txt = r'C:\Users\suman\Downloads\DCNPOJECT\DCNPOJECT\SMSSpamCollection'
output_file_path_csv = r'C:\Users\suman\Downloads\DCNPOJECT\DCNPOJECT\SMSSpamCollection.csv'

# Convert TXT to CSV
def convert_txt_to_csv(txt_path, csv_path):
    df = pd.read_csv(txt_path, sep='\t', header=None, names=['Label', 'Message'])
    df.to_csv(csv_path, index=False, header=True)
    print(f"CSV file created: {csv_path}")

# Convert the TXT file to CSV
convert_txt_to_csv(input_file_path_txt, output_file_path_csv)

# Load the CSV dataset
df = pd.read_csv(output_file_path_csv)

# Convert labels to binary (1 for spam, 0 for not spam)
df['Label'] = df['Label'].map({'spam': 1, 'ham': 0})

# Prepare the messages and labels
messages = df['Message']
labels = df['Label']

# Convert text data into numerical data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create classifiers
nb = MultinomialNB()
svm = SVC(kernel='linear', probability=True)
log_reg = LogisticRegression()

# Create a voting classifier (ensemble)
ensemble_model = VotingClassifier(estimators=[
    ('naive_bayes', nb), 
    ('svm', svm), 
    ('log_reg', log_reg)
], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Function to detect phishing
def is_phishing(message):
    phishing_keywords = ["urgent", "account", "verify", "click", "password", "bank"]
    message = message.lower()
    return any(keyword in message for keyword in phishing_keywords)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    message = request.form['message']
    if not message:
        result = "Please enter a message to classify."
    else:
        # Convert the input message to the format expected by the model
        message_vector = vectorizer.transform([message])
        is_spam = ensemble_model.predict(message_vector)[0]
        
        # Check if the message is phishing
        if is_phishing(message):
            result = "Phishing detected! Message will be deleted."
        elif is_spam:
            result = "Spam detected!"
        else:
            result = "Not Spam"
        
    return render_template('result.html', message=message, result=result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
