# 💼 Job Salary Prediction with Text Analytics

This repository applies **Natural Language Processing (NLP)** and machine learning to predict whether a job posting corresponds to a **high** or **low salary**, based on the full job description.  
The project demonstrates text preprocessing, feature engineering, classification, and interpretability using word importance.

---

## 📂 Project Structure
- `Individual_Assignment-Julien_Palummo-v2.py` → Python script implementing the full workflow (cleaning, training, evaluation, visualization).  
- `Individual_Assignment-Julien_Palummo-v2.pdf` → Final report with methodology, results, and discussion.  

---

## ⚙️ Workflow & Features

### 1. Data Preparation
- Loaded and sampled 2,500 job descriptions.  
- Cleaned text:
  - Lowercasing, URL & punctuation removal, lemmatization (SpaCy).  
  - Removed high-frequency terms with `max_df=0.2`.  
- Defined salary classes:
  - **High salary (1):** above 75th percentile of normalized salary.  
  - **Low salary (0):** otherwise.  

### 2. Model Training
- Used **Bag-of-Words** features with `CountVectorizer`.  
- Trained a **Naïve Bayes classifier (MultinomialNB)** in a pipeline.  
- Split into **80% training / 20% testing**.  

### 3. Performance
- **Accuracy:** ~0.80   
- **Confusion Matrix:**  
  - Stronger performance for low-salary predictions.  
  - Some imbalance in high-salary predictions due to class skew.  

### 4. Interpretability
Top words indicative of each salary class:  

- **High salary:** marketing, financial, software, developer, senior, engineer, solution, product, technical, design.  
- **Low salary:** hour, maintain, standard, people, engineer, product, staff, account, design, care .  

Visualizations include:  
- Accuracy bar chart.  
- Confusion matrix heatmap.  
- Bar plots of top 10 predictive words per class.  

### 5. Improvements Considered
- Use **n-grams (bigrams/trigrams)** to capture context.  
- Add **structured features** like job title, location, company size.  
- Handle class imbalance with **SMOTE** or class weights .  

---

## 📈 Results & Insights
- Even simple text models can achieve ~80% accuracy.  
- Certain keywords strongly correlate with salary levels.  
- Class imbalance remains a key limitation for predicting high salaries.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** pandas, scikit-learn, spaCy, matplotlib, seaborn  
- **Techniques:** Text cleaning, Bag-of-Words, Naïve Bayes, confusion matrix analysis, keyword interpretability  
