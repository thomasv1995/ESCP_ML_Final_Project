# Predicting Age and Gender Based on Search Query Keywords

*Group 9: Amélie Pingeot, Taanya Sachdeva, Maxime Pracisnore, Maximilian Baum, Thomas Vermaelen*  

## Project Outline

### I. Data Preparation

• Sample 50,000 obseervations
• check distribution of classes within the target variable "sex": looks fine
• Clean, tokenize and stem keywords

### II. Modeling Gender 

#### A) Tf-Idf with Full Vocabulary

• build vocabulary using Tf-Idf : length of vocabulary is 31,066 words  
• Vectorize tokens to represent them numerically using Tf-Idf  
• Perform Logistic Regression:  
    - AUC Train (5-fold CV): 63.8%  
    - AUC Test: 61.3%  

#### B) Tf-Idf with Reduced Vocabulary (Nouns and Verbs only)

#### C) Word2Vec

#### D) Neural Network (with Tf-Idf Embeddings)

### III. Predicting Gender With Multinomial Naive Bayes and Tf-Idf Embeddings

### IV. Modeling Age

#### A) Tf-Idf Full Vocabulary

#### B) Tf-Idf Reduced Vocabulary (Nouns and Verbs only)

#### C) Word2Vec

### V. Predicting Age with Linear Regression and Word2Vec
