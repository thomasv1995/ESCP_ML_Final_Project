# Predicting Age and Gender Based on Search Query Keywords

*Group 9: Amélie Pingeot, Taanya Sachdeva, Maxime Pracisnore, Maximilian Baum, Thomas Vermaelen*  

This is the final project for our class "Machine Learning in Python". This project was assigned to us by Le Figaro (a French news agency) in which we are required to predict the age and gender of users based on their search queries on Le Figaro website. We find that the best model for predicting Gender is a Multinomial Naive Bayes model with Tf-Idf embeddings for the keywords (Accuracy = 58%, AUC = 61.4%). When predicting age, we have found that Linear Regression with Word2Vec embeddings yielded the best result (RMSE = 13.02).    

## Project Outline

### I. Data Preparation

• Sample 50,000 obseervations   
• check distribution of classes within the target variable "sex": looks fine  
• Clean, tokenize and stem keywords  


### II. Modeling Gender 

• Peform 80/20 training and validation split  
• Scale data 

#### A) Tf-Idf with Full Vocabulary

• Build vocabulary using Tf-Idf : length of vocabulary is 31,066 words  
• Vectorize tokens to represent them numerically using Tf-Idf  
• Perform Logistic Regression:  
    - AUC Train (5-fold CV): 63.8%  
    - AUC Test: 61.3%  
• Multionmial Naive Bayes:  
    - AUC Train (5-fold CV): 64.1%  
    - AUC Test: 61.4%  

#### B) Tf-Idf with Reduced Vocabulary (Nouns and Verbs only)

• Build Tf-Idf vocabulary retaining only nouns and verbs this time: length of vocabulary is 20,055 words. 
• Vectorize tokens  
• Logistic Regression:  
    - AUC Train (5-fold CV): 63.5%  
    - AUC Test: 61%  
• Multionmial Naive Bayes:  
    - AUC Train (5-fold CV): 63.6%  
    - AUC Test: 61.2%  

#### C) Word2Vec

• Use Word2Vec to create vector embeddings for keyword tokens
• Logistic Regression:  
    - AUC Train (5-fold CV): 55.3%  
    - AUC Test: 55.2%  
• Multionmial Naive Bayes:  
    - AUC Train (5-fold CV): 58.2%  
    - AUC Test: 57.3%  

#### D) Neural Network (with Tf-Idf Embeddings)

• Use Dense Neural Network with Tf-Idf embeddings to predict gender  
• AUC Train: 61.2%  
• AUC Test: 57.6%  

**Conclusion: Multinomial Naive Bayes with Tf-Idf Embeddings (Full Vocabulary) yielded the best testing metrics (Accuracy = 58%, AUC = 61.4%), so we will use it to predict "Sex" for the provided test dataset**  

### III. Predicting Gender With Multinomial Naive Bayes and Tf-Idf Embeddings

• We use our best model (i.e. MNB with Tf-Idf embeddings) to predict gender on a testing set that was provided

### IV. Modeling Age

#### A) Tf-Idf Full Vocabulary

• Linear Regression scores (Tf-idf full vocab):  
    - Root Mean Squared Error: 8.62598807554911  
    - Root Mean Squared Error (Test): 63.72276579663561  

#### B) Tf-Idf Reduced Vocabulary (Nouns and Verbs only)

Random Forest scores (Tf-idf full vocab)):  
    - Root Mean Squared Error: 8.04264104632303  
    - Root Mean Squared Error (Test): 16.542342639420813  

#### C) Word2Vec

Linear Regression (Tf-idf new vocab):
    - Root Mean Squared Error: 9.902500164418019  
    - Root Mean Squared Error (Test): 89.5590398559217  

**Conclusion: Linear regression with Word2Vec yielded the lowest testing RMSE (13.02), therefore, we will use this model to predict age.**  

### V. Predicting Age with Linear Regression and Word2Vec

We use our best model (i.e. Linear regression with Word2Vec) to predict age on a given testing set  

