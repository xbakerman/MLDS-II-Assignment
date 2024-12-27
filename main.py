import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score, f1_score,
                             classification_report)


################################ FUNCTIONS######################################

def read_email_from_folder(folder_path):
    """Reads all emails from a folder and returns a list of emails.

    An empty list is initialized to store the emails.
    The function iterates through all files in the specified folder.
    For each file, it reads the content of the file and appends it to the
    emails list.
    The function returns the list of emails.

    Args:
        folder_path (str): Path of the folder containing the emails.

    Returns:
        list: A list containing the emails.

    """
    # Reads the emails from the folder and stores them in a list
    emails = []

    # Iterates over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="latin-1") as file:
            email = file.read()
            emails.append(email)

    return emails


def preprocess_emails(email):
    """Preprocesses the email by removing unwanted characters and converting it
    to lowercase.

    The preprocessing steps include removing headers, URLs, email addresses,
    numbers, punctuation, and special characters. It also standardizes the
    email content for further processing.

    re.sub() is a function from the re module that replaces substrings that
    match a pattern with a new substring.
    For different patterns and replacements, the re.sub() function is called
    multiple times.

    Args:
        email (str): The raw email to preprocess.

    Returns:
        str: The preprocessed email.
    """
    # Removes URL and replaces with "URL"
    email = re.sub(r"http\S+|www\S+|https\S+", "URL", email)

    # Removes HTML tags
    email = re.sub(r"<[^>]*>", "", email)

    # Converts to lowercase
    email = email.lower()

    # Removes Email and replaces with "EMAIL"
    email = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "EMAIL", email)

    # Removes Numbers and replace with "NUMBER"
    email = re.sub(r"\d+", "NUMBER", email)

    # Removes Punctuation and special characters and removes words with more
    # than 18 characters
    email = re.sub(r"[^a-z\s]", "", email)
    email = re.sub(r"\b[a-z]{18,}\b", "", email)

    # Removes non-word characters and replaces with space
    email = re.sub(r"\W+", " ", email)

    # Removes extra spaces
    email = re.sub(r"\s+", " ", email).strip()

    return email


def process_emails(folder_path):
    """Reads and processes all emails from a folder. 

    The function reads the emails from the specified folder and processes them
    using the preprocess_emails() function.
    The processed emails are stored in a list and returned.

    Args:
        folder_path (str): The path to the folder containing the emails.

    Returns:
        list: A list containing the processed emailas.
    """
    # Reads the emails from the folder and process them
    raw_emails = read_email_from_folder(folder_path)
    processed_emails = [preprocess_emails(email) for email in raw_emails]

    return processed_emails


def vectorizing_emails(emails, count=True, vocab=None):
    """Vectorizes the emails using the Tf-idf Vectorizer.

    The function vectorizes the emails using the TfidfVectorizer from the 
    scikit-learn library.
    The vectorizer is fit on the emails and the feature matrix is generated.

    The TfidVectorizer converts the emails into a sparse matrix representation,
    where each row corresponds to an email and each column corresponds to a 
    word in the vocabulary. The value in each cell represents the TF-IDF score 
    of the word in the email. 

    Args:
        emails (list): A list containing the cleaned emails.
        count (bool): A boolean indicating whether to use count or binary
                      vectorization.
        vocab (list): A list containing the vocabulary.

    Returns:
        tuple: A tuple containing the feature matrix and the vectors
               containing the vocabulary in numeric form.
    """

    # Vectorizes the emails using the TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=10000, max_df=0.5, min_df=1, stop_words="english")
    vectors = vectorizer.fit_transform(emails)

    return vectors, vectorizer


def analyze_word_frequencies(vectors, labels, vocab):
    """Analyzes the word frequencies in the emails (both spam and 
    not-spam emails).

    The function calculates the word frequencies for both spam and not-spam
    emails and combines them into a single dataframe.
    The word frequencies are sorted in descending order based on the total
    frequency in both spam and not spam.


    Args:
        emails (list): A list containing the clened emails.
        labels (list): A list containing the labels of the emails 
                       (1 = Spam, 0 = Not Spam).

    Returns:
        Dataframe: A dataframe containing the word frequencies.
    """
    # Creates a dataframe containing the word counts
    word_counts = pd.DataFrame(vectors.toarray(), columns=vocab)
    word_counts["Label"] = labels

    # Calculates the word counts for spam and ham emails
    spam_word_counts = word_counts[word_counts["Label"] == 1].drop(
        columns="Label").sum()
    ham_word_counts = word_counts[word_counts["Label"] == 0].drop(
        columns="Label").sum()

    # Combines the word counts into a single dataframe
    word_counts = pd.DataFrame(
        {"Spam": spam_word_counts, "Ham": ham_word_counts})
    word_counts["Total"] = word_counts["Spam"] + word_counts["Ham"]

    return word_counts.sort_values(by="Total", ascending=False)


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                             model_name="Model"):
    """Trains and evaluates a model on the training and testing sets for
    different models.

    The function trains the model on the training set and evaluates it on the
    testing set using the accuracy, precision, recall, and F1 score.
    The evaluation metrics are printed to the console.

    Args:
        model: The model to train and evaluate.
        X_train (array): The feature matrix of the training set.
        y_train (array): The labels of the training set.
        X_test (array): The feature matrix of the testing set.
        y_test (array): The labels of the testing set.

    Returns:
        None
    """
    # Trains the model on the training set
    model.fit(X_train, y_train)

    # Predicts the labels of the test set
    y_pred = model.predict(X_test)

    # Calculates the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Prints the evaluation metrics
    print(f"{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f} \n")
    print(f'Classification Report for {model_name}: \n {
          classification_report(y_test, y_pred)} \n')


def plot_confusion_matrix(models, X_test, y_test, labels=["Ham", "Spam"]):
    """Plots confusion matrices for all models. 

    The function plots the confusion matrices for each model in the list
    of models.
    The confusion matrix shows the true positive, true negative, 
    false positive, and false negative values for each model.


    Args:
        models (list of tuples): A list of tuples where each tuple contains
                                 the model name and model instance.
        X_test (array): The test feature matrix.
        y_test (array): The true labels for the test set.
        labels (list): The class labels for the confusion matrix 
                       (default: ["Ham", "Spam"]).

    Returns:
        None
    """
    # Calculates number of models and number of rows and columns for subplots
    n_models = len(models)
    cols = 3
    rows = math.ceil(n_models / cols)

    # Creates subplots
    fig, axes = plt.subplots(rows, cols, figsize=(10, 6))

    # Flattens axes for easier indexing
    axes = axes.flatten()

    for i, (model_name, model) in enumerate(models):
        # Predicts the test set
        y_pred = model.predict(X_test)

        # Computes the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plotting the confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                    xticklabels=labels, yticklabels=labels, ax=axes[i])
        axes[i].set_title(f"{model_name}")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")

    # Hides unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  

    plt.tight_layout()
    plt.show()


######################### LOADING AND PROCESSING DATA##########################
'''The emails are divided into three categories: spam, easy ham, and hard ham.

The Data gets read from the folders and processed with the built functions 
above.
'''
# Paths to the folders containing the emails
spam_mail_folder = "Datasets/spam"
easy_ham_mail_folder = "Datasets/easy_ham"
hard_ham_mail_folder = "Datasets/hard_ham"
spam_2_mail_folder = "Datasets/spam_2"
easy_ham_2_mail_folder = "Datasets/easy_ham_2"

# Reads the emails from the folders
spam_mails = read_email_from_folder(spam_mail_folder)
spam2_mails = read_email_from_folder(spam_2_mail_folder)
easy_ham_mails = read_email_from_folder(easy_ham_mail_folder)
easy_ham2_mails = read_email_from_folder(easy_ham_2_mail_folder)
hard_ham_mails = read_email_from_folder(hard_ham_mail_folder)

# Processes the emails
processed_spam_mails = process_emails(
    spam_mail_folder) + process_emails(spam_2_mail_folder)
processed_easy_ham_mails = process_emails(
    easy_ham_mail_folder) + process_emails(easy_ham_2_mail_folder)
processed_hard_ham_mails = process_emails(hard_ham_mail_folder)


######################### VECTORIZING EMAILS###################################
'''The processed emails are combined and the labels are created.

The emails are vectorized and the vocabulary size, vocabulary and feature 
matrix dimension are extracted.

To get an idea of the most common words in the emails, the word frequencies 
are analyzed.
'''
# Combines all emails
all_emails = processed_spam_mails + \
    processed_easy_ham_mails + processed_hard_ham_mails

# Creates labels for the emails: 1 = spam, 0 = ham
all_labels = [1] * len(processed_spam_mails) + [0] * \
    (len(processed_easy_ham_mails) + len(processed_hard_ham_mails))

# Cleans and vectorizes the emails and print the vocabulary size,
# vocabulary, feature matrix dimension
vectors, vectorizer = vectorizing_emails(all_emails)
vocab = vectorizer.get_feature_names_out()


######################### WORD FREQUENCY ANALYSIS##############################
'''To use only the most relevant words, the words with the highest 
difference in frequency between spam and ham emails are selected.

The emails are then vectorized again using only the relevant words.
'''
# Analyzes the word frequencies in the emails to identify the most common words
word_frequencies = analyze_word_frequencies(vectors, all_labels, vocab)

# Words with the highest difference in frequency between spam and ham emails
word_frequencies["Difference"] = abs(
    word_frequencies["Spam"] - word_frequencies["Ham"])
relevant_words = word_frequencies.sort_values(
    by="Difference", ascending=False).index[:10000]

# Cleans and vectorizes the emails again using the relevant words
vectors, vectorizer = vectorizing_emails(all_emails, vocab=relevant_words)


######################### SHUFFLING AND SPLITTING DATA#########################
'''The data is shuffled and split into training and testing sets. '''

# Shuffles the data
vectors, all_labels = shuffle(vectors, all_labels, random_state=42)

# Splits the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    vectors, all_labels, test_size=0.3, random_state=42)


######################### MODEL TRAINING AND EVALUATION########################
'''Different models are trained and evaluated on the training and testing sets.

To get a better understanding of the model performance, the confusion matrix is
plotted for each model.
'''

# Defines the models to train and evaluate
models = [
    ("Naive Bayes", MultinomialNB(alpha=0.001)),
    ("Logistic Regression", LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42)),
    ("Random Forest", RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42)),
    ("Linear SVM", LinearSVC(C=1, class_weight="balanced", random_state=42)),
    ("SGD Classifier", SGDClassifier(alpha=0.001,
     class_weight="balanced", random_state=42))
]

# Trains and evaluates the models
for model_name, model in models:
    train_and_evaluate_model(model, X_train, y_train,
                             X_test, y_test, model_name=model_name)

# Plotting confusion matrix for each model
plot_confusion_matrix(models, X_test, y_test)
