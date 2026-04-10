# Fake News Detection using NLP & Machine Learning

This project builds a machine learning model to classify news articles as Real or Fake using Natural Language Processing (NLP) techniques.
[Dataset link](https://www.kaggle.com/datasets/subhajournal/fake-and-real-news-data)

## Key Highlights

- Built an end-to-end NLP pipeline using scikit-learn
- Achieved ~93% accuracy using LinearSVC
- Compared multiple baseline models
- Implemented TF-IDF vectorization
- Applied hyperparameter tuning with GridSearchCV

## Tech Stack
Pandas, Scikit-learn, Nltk

## Methodology

1. Data Preprocessing
  - Used text column
  - Cleaned text (removed special characters)
  - Converted to lowercase
  - Removed stopwords

2. Feature Engineering
  - Used TF-IDF Vectorization

3. Models Tested
  - Logistic Regression
  - Multinomial Naive Bayes
  - LinearSVC (Best Performer)
  - Model pipeline:
    ```python
    preprocess = Pipeline(
        steps = [
            ('tfidf',TfidfVectorizer())
        ]
	)
	base_pipeline = Pipeline(
    steps=[
        ('preprocess',preprocess),
        ('model',LinearSVC())
    ]
	)
    ```

 4. Model Evaluation
  - Logistic Regression: ~91%
  - Naive Bayes: ~84%
  - LinearSVC: ~93%

5. Hyperparameter Tuning
  - Performed GridSearchCV on LinearSVC:
    
    ```python
     param_grid = {
    'model__C': [0.1, 0.5, 1, 2, 5],
    'model__loss': ['hinge', 'squared_hinge'],
    'model__max_iter': [2000, 5000],
    'model__class_weight': [None, 'balanced']
    }
    ```

## Results
- Best Model: LinearSVC
- Final Accuracy: ~93%
- The model successfully captures linguistic patterns distinguishing fake and real news.

## How to Clone this Project

1. Clone the repository:
   ```bash
   git clone https://github.com/aayushmanmukherjee/Fake_News_Detection_NLP.git
   ```
2. Install Python libraries in a virtual environment:

   - For Mac/Linux
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   - For Windows
    ```bash
   python3 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
    
3. Run the `.ipynb` file
