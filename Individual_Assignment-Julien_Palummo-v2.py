import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
import numpy as np
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns


# Load Spacy English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Define the text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove newline characters
    text = re.sub(r'\n', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove user @ references and '#' symbols
    text = re.sub(r'@\w+|#', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Lemmatization
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# Load the dataset
file_path = 'C:/Users/julie/OneDrive/Documents/MMA/Winter 2024/Text Analytics/job-salary-prediction (1)/Train_rev1/Train_rev1.csv'
data = pd.read_csv(file_path)

# Randomly select 2500 data points from the dataset
sample_data = data.sample(n=2500, random_state=42)

# Preprocess the FullDescription column
sample_data['CleanedDescription'] = sample_data['FullDescription'].apply(clean_text)

# Split the data into training and test sets (80% training, 20% test)
train_data, test_data = train_test_split(sample_data, test_size=0.2, random_state=42)

# Define high and low salary based on the 75th percentile of the SalaryNormalized column in the sample data
salary_threshold = sample_data['SalaryNormalized'].quantile(0.75)
train_data['SalaryClass'] = (train_data['SalaryNormalized'] > salary_threshold).astype(int)
test_data['SalaryClass'] = (test_data['SalaryNormalized'] > salary_threshold).astype(int)

# Text preprocessing and feature extraction
vectorizer = CountVectorizer(stop_words='english', max_df=0.2)

# Build a Naïve Bayes classifier pipeline
model = make_pipeline(vectorizer, MultinomialNB())

# Train the model
model.fit(train_data['CleanedDescription'], train_data['SalaryClass'])

# Predict on the test set
predictions = model.predict(test_data['CleanedDescription'])

# Calculate accuracy
accuracy = accuracy_score(test_data['SalaryClass'], predictions)

# Generate confusion matrix
conf_matrix = confusion_matrix(test_data['SalaryClass'], predictions)

# Get feature names from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Get the class log probabilities
log_probabilities = model.named_steps['multinomialnb'].feature_log_prob_

# Identify the top 10 words for each class
# High salary class (1)
top_words_high_salary = np.argsort(log_probabilities[1])[-10:]
# Low salary class (0)
top_words_low_salary = np.argsort(log_probabilities[0])[-10:]

# Map indices to words
top_words_high_salary = [feature_names[i] for i in top_words_high_salary]
top_words_low_salary = [feature_names[i] for i in top_words_low_salary]

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Top 10 Words for High Salary:", top_words_high_salary)
print("Top 10 Words for Low Salary:", top_words_low_salary)

# Set the style of seaborn
sns.set_style("whitegrid")

# Plot Accuracy Score
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.title('Model Accuracy Score')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  
plt.show()

# Visualizing the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Low Salary', 'High Salary'], yticklabels=['Low Salary', 'High Salary'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Function to plot top words for each class
def plot_top_words(class_words, class_title):
    words = class_words[::-1]
    plt.figure(figsize=(10, 8))
    sns.barplot(x=np.arange(1, 11), y=words, palette='coolwarm')
    plt.title(f'Top 10 Words for {class_title}')
    plt.ylabel('Words')
    plt.xlabel('Importance')
    plt.show()

# Plotting top words for each class
plot_top_words(top_words_high_salary, "High Salary")
plot_top_words(top_words_low_salary, "Low Salary")


