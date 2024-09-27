from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier  # Import for Decision Tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import io
import base64
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# Function to provide research-based model recommendations for different domains
def research_based_recommendations(domain):
    domain_models = {
        "Finance": "RandomForest",
        "Healthcare": "NeuralNetwork",
        "Retail": "RandomForest",
        "Education": "NeuralNetwork",
        "Manufacturing": "SVM",
        "Telecommunications": "SVM",
        "Transportation": "RandomForest",
    }
    return domain_models.get(domain, "No specific model recommendation available for this domain.")

# Function to load and preprocess data
def load_and_preprocess_data(df):
    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine the transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y

# Function to evaluate model performance
def evaluate_model(y_test, y_pred, y_pred_prob=None):
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
    }
    if y_pred_prob is not None:
        metrics["AUC-ROC"] = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    return metrics

# Function to plot pie chart for model performance
def plot_pie_chart(metrics):
    labels = metrics.keys()
    sizes = metrics.values()
    colors = ['#66b3ff', '#ff6666', '#99ff99', '#ffcc99']

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
# Function to train traditional machine learning models
def train_traditional_models(X_train, y_train, X_test, y_test):
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),  # Add Decision Tree
        "KNN": KNeighborsClassifier()  # Add KNN
    }
    results = {}
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            metrics = evaluate_model(y_test, y_pred, y_pred_prob)
            pie_chart = plot_pie_chart(metrics)
            results[model_name] = {"metrics": metrics, "pie_chart": pie_chart}
        except Exception as e:
            results[model_name] = {"metrics": {"Error": str(e)}, "pie_chart": None}
    return results



# Function to train a neural network
def train_neural_network(X_train, y_train, X_test, y_test):
    input_dim = X_train.shape[1]
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(np.unique(y_train_encoded))

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))

    if num_classes > 2:
        model.add(Dense(num_classes, activation='softmax'))
        y_train_cat = to_categorical(y_train_encoded, num_classes=num_classes)
        loss_function = 'categorical_crossentropy'
    else:
        model.add(Dense(1, activation='sigmoid'))
        y_train_cat = y_train_encoded
        loss_function = 'binary_crossentropy'

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=0, validation_split=0.1)

    y_pred_prob = model.predict(X_test)

    if num_classes > 2:
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    metrics = evaluate_model(y_test_encoded, y_pred)
    pie_chart = plot_pie_chart(metrics)

    return {"NeuralNetwork": {"metrics": metrics, "pie_chart": pie_chart}}

# Function to select the best model based on performance metrics
def select_best_model(results):
    return max(results, key=lambda model: results[model]["metrics"].get("Accuracy", 0))

# Function to create a CSV for download
def create_results_csv(results):
    df = pd.DataFrame({model: data["metrics"] for model, data in results.items()}).T
    df.to_csv('model_results.csv', index=True)
    return 'model_results.csv'

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username is an email and password is "admin"
        if '@' in username and username.count('@') == 1 and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return 'Invalid credentials, please try again.'
    return render_template('login.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        domain = request.form['domain']
        file = request.files['dataset']

        if file:
            try:
                df = pd.read_csv(file)
                X, y = load_and_preprocess_data(df)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train models
                traditional_results = train_traditional_models(X_train, y_train, X_test, y_test)
                neural_network_results = train_neural_network(X_train, y_train, X_test, y_test)

                # Combine results
                combined_results = {**traditional_results, **neural_network_results}

                # Select best model based on accuracy
                best_model = select_best_model(combined_results)

                # Get domain-specific recommendation
                domain_recommendation = research_based_recommendations(domain)

                # Create CSV for download
                results_csv = create_results_csv(combined_results)

                return render_template('result.html', results=combined_results, best_model=best_model, 
                                       domain_recommendation=domain_recommendation, csv_file=results_csv)

            except Exception as e:
                return f"Error processing the dataset: {str(e)}"
        else:
            return "No file uploaded. Please upload a dataset."
    
    return render_template('index.html')

@app.route('/download')
def download():
    return send_file('model_results.csv', as_attachment=True)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)

