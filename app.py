
from flask import Flask, render_template, request, Response,make_response
import joblib
import os
import base64
import io
import matplotlib
import sqlite3
from collections import Counter
from flask import g
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import g
from database import get_db
from flask import redirect, request
from pandas.errors import ParserError
from sklearn.pipeline import Pipeline
import csv


matplotlib.use('Agg')

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['TEMPLATE_FOLDER'] = os.path.abspath('templates')
app.config['DATABASE'] = 'messages.db'

classifiers = {
    'Random Forest': {
        'model_path': 'E:/python/RandomForestClassifier_CountVectorizer_model.joblib',
        'vectorizer_path': 'E:/python/RandomForestClassifier_CountVectorizer_vectorizer.joblib'
    },
    'Naive Bayes': {
        'model_path': 'E:/python/NaiveBayes_CountVectorizer_model.joblib',
        'vectorizer_path': 'E:/python/NaiveBayes_CountVectorizer_vectorizer.joblib'
    },
    'Logistic Regression': {
        'model_path': 'E:/python/LogisticRegression_CountVectorizer_model.joblib',
        'vectorizer_path': 'E:/python/LogisticRegression_CountVectorizer_vectorizer.joblib'
    }
}

selected_classifier = 'Random Forest'
##model = joblib.load(classifiers[selected_classifier]['model_path'])
##vectorizer = joblib.load(classifiers[selected_classifier]['vectorizer_path'])
pipeline = joblib.load(classifiers[selected_classifier]['model_path'])


X_test = ['This is a test message.', 'Another test message.', 'A third test message.']
y_test = [0, 1, 0]

DATABASE = 'database.db'

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def create_tables():
    conn = get_db()
    conn.execute('DROP TABLE IF EXISTS saved_messages')
    conn.execute('DROP TABLE IF EXISTS imported_messages')
    conn.execute('''CREATE TABLE IF NOT EXISTS imported_messages (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   message TEXT,
                   predicted_label TEXT,
                   classifier TEXT)''')
    conn.commit()



@app.route('/import-messages', methods=['GET', 'POST'])
def import_messages():
    selected_classifier = 'Random Forest'  # Set a default value
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # Check if the file has a valid extension
        if not file.filename.endswith('.csv'):
            return "Invalid file extension. Only CSV files are supported."

        try:
            df = pd.read_csv(file, encoding='latin-1')
        except pd.errors.ParserError as e:
            return f"Error parsing CSV file: {str(e)})"

        conn = get_db()
        conn.execute('DELETE FROM imported_messages')

        selected_classifier = request.form.get('classifier')

        # Classify and insert each message into the database
        for index, row in df.iterrows():
            message = row['text']
            predicted_label = classify_message(message)  # Perform classification here
            conn.execute('INSERT INTO imported_messages (message, predicted_label, classifier) VALUES (?, ?, ?)',
                         (message, predicted_label, selected_classifier))

        conn.commit()

        # Fetch all imported messages from the database
        cur = conn.cursor()
        cur.execute('SELECT message, predicted_label, classifier FROM imported_messages')
        imported_messages = cur.fetchall()

        return render_template('imported_messages.html', imported_messages=imported_messages, classifiers=classifiers)

    return render_template('import_messages.html', classifiers=classifiers, selected_classifier=selected_classifier)





@app.route('/imported-messages')
def imported_messages():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM imported_messages')
    messages = cur.fetchall()
    return render_template('imported_messages.html', messages=messages, classifiers=classifiers, selected_classifier=selected_classifier)



    
@app.route('/download-messages')
def download_messages():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT message, predicted_label, classifier FROM imported_messages')
    imported_messages = cur.fetchall()

    # Create a CSV file with the imported messages
    csv_data = "Message,Predicted Label,Classifier\n"
    for message in imported_messages:
        csv_data += f"{message[0]},{message[1]},{message[2]}\n"

    # Create a response with the CSV file data
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=imported_messages.csv"
    response.headers["Content-type"] = "text/csv"

    return response

     

def insert_message(message, predicted_label):
    conn = get_db()
    conn.execute('INSERT INTO messages (message, predicted_label) VALUES (?, ?)', (message, predicted_label))
    conn.commit()

def get_messages():
    conn = get_db()
    messages = conn.execute('SELECT * FROM messages').fetchall()
    return messages

def get_analytics():
    conn = get_db()
    messages = conn.execute('SELECT message FROM messages').fetchall()
    messages = [msg[0] for msg in messages]

    if not messages:
        return {
            'num_messages': 0,
            'avg_length': 0,
            'common_words': []
        }

    message_lengths = [len(msg.split()) for msg in messages]
    words = [word for msg in messages for word in msg.split()]
    common_words = Counter(words).most_common(10)

    return {
        'num_messages': len(messages),
        'avg_length': round(sum(message_lengths) / len(messages), 2),
        'common_words': common_words
    }

def evaluate_model(model_path, test_data_path):
    test_data = pd.read_csv(test_data_path)
    model = joblib.load(model_path)

    test_features = test_data['text']
    test_labels = test_data['label']
    test_labels = test_labels.map({'ham': 0, 'spam': 1})

    predictions = model.predict(test_features)
    predictions = np.where(predictions == 'spam', 1, 0)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    return accuracy, precision, recall, f1

@app.route('/')
def home():
    messages = get_messages()
    analytics = get_analytics()
    return render_template('home.html', messages=messages, analytics=analytics, classifiers=classifiers, selected_classifier=selected_classifier)


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    classifier = request.form['classifier']

    model = joblib.load(classifiers[classifier]['model_path'])
    vectorizer = joblib.load(classifiers[classifier]['vectorizer_path'])

    proba = model.predict_proba([message])[0]
    predicted_label = model.classes_[proba.argmax()]
    prob_spam = round(proba[1]*100, 2)
    prob_ham = round(proba[0]*100, 2)

    fig, ax = plt.subplots()
    ax.bar(['Ham', 'Spam'], [prob_ham, prob_spam], color=['green', 'red'])
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 100])
    ax.set_title('Spam Classification Result')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()

    insert_message(message, predicted_label)

    return render_template('result.html', message=message, label=predicted_label, prob_spam=prob_spam, prob_ham=prob_ham, plot_data=plot_data, analytics=get_analytics(), classifiers=classifiers, selected_classifier=classifier)


@app.route('/messages')
def messages():
    messages = get_messages()
    return render_template('messages.html', messages=messages, classifiers=classifiers, selected_classifier=selected_classifier)


@app.route('/analytics')
def analytics():
    analytics = get_analytics()
    return render_template('analytics.html', analytics=analytics, classifiers=classifiers, selected_classifier=selected_classifier)

@app.route('/evaluation', methods=['GET', 'POST'])
def evaluation():
    if request.method == 'POST':
        selected_classifier = request.form.get('classifier')
        if selected_classifier in classifiers.keys():
            model_path = classifiers[selected_classifier]['model_path']
            test_data_path = 'E:/python/SMSSpamCollection.csv'

            accuracy, precision, recall, f1 = evaluate_model(model_path, test_data_path)

            return render_template('evaluation.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, selected_classifier=selected_classifier, classifiers=classifiers)
        else:
            return redirect('/evaluation')  # Redirect to the evaluation page if an invalid classifier is selected
    else:
        return render_template('evaluation.html', accuracy='', precision='', recall='', f1='', classifiers=classifiers)

def classify_message(message):
    # Vectorize the message using the same vectorizer used during training
    ##vectorized_message = vectorizer.transform([message])

    # Predict the label using the trained model
    predicted_label = pipeline.predict([message])[0]

    return predicted_label

if __name__ == '__main__':
    with app.app_context():
        conn = get_db()
        create_tables()  # Add this line to create the tables
        with app.open_resource('database.sql', mode='r') as f:
            conn.cursor().executescript(f.read())
    app.run(debug=True, host="0.0.0.0", port=5000)
