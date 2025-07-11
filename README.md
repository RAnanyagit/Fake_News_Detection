# Fake News Detection using Machine Learning

This project detects whether a news article is Real or Fake using NLP (Natural Language Processing) techniques and a Logistic Regression model. Built using Python and Google Colab for quick experimentation.

## Tech Stack

- Python 
- Pandas, NumPy for data handling
- Scikit-learn (TF-IDF, Logistic Regression)
- Matplotlib, WordCloud (optional for visualization)
- Google Colab 

##  Dataset

- Merged dataset with two files: `True.csv` and `Fake.csv`
- Contains thousands of real and fake news articles with:
  - title
  - text
  - label (0 = Fake, 1 = Real)
## Features Implemented
1.Data Cleaning
Cleaned the news content by removing special characters, punctuation, converting to lowercase, and removing extra spaces.

2.Combined Title & Text
Merged the title and text columns into a single content column for better context and accuracy.

3.TF-IDF Vectorization
Used TfidfVectorizer to convert text data into numerical format, giving weight to important words.

4.Logistic Regression Model
Trained a Logistic Regression model to classify whether the news is real or fake.

5.Model Evaluation
Evaluated the model using a confusion matrix and classification report (precision, recall, F1-score, accuracy).

 6.Custom Prediction
Added a function check_news() to test custom input and check if it’s real or fake

## How to Run

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Upload both `True.csv` and `Fake.csv`
3. Run all cells in order

## Sample Output
Confusion Matrix:
[[4641    9]
 [  13 4317]]


Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      4650
           1       1.00      1.00      1.00      4330

    accuracy                           1.00      8980
   macro avg       1.00      1.00      1.00      8980
weighted avg       1.00      1.00      1.00      8980

Custom Prediction Example:
check_news("Government confirms new economic reforms")

