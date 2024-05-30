# Sentiment Analysis from Tweets

## Introduction
This project utilizes a Support Vector Machine (SVM) to classify tweets into positive or negative sentiments. Given the proliferation of social media data, understanding these sentiments provides valuable insights into public opinion. This README outlines the methodology, structure, and outcomes of the project, focusing on machine learning techniques for natural language processing (NLP).

## Dataset Description
The dataset comprises over 30,000 tweets, each annotated with a sentiment label (positive or negative). These tweets cover various topics, providing a diverse corpus for training and testing the classifier. The data is sourced from a tab-separated values file (`sentiment-dataset.tsv`), with each record containing the tweet ID, sentiment label, and text.

### Sample Data
Here are a few examples from the dataset:
```
410734138242126311 positive Literally so excited I'm going to a Sam Smith concert in October
450236582392850660 negative @AngryRaiderFan I know. This, TPP, expanded wars and drone strikes, mass surveillance, on and on...
686031506093676865 positive @rinashah I have been using Moto G 2nd Gen for over a month now and it's an absolute delight. Stock Android. Good design. Best.
```


### Project File Structure
This project is organized into two primary Python files that reflect the stages of development and optimization of the sentiment analysis model:

1. **Sentiment_analysis_from_tweets_before_preprocessing.ipynb**
   - **Purpose**: This file contains the initial stages of the project, including data input, basic preprocessing, simple feature extraction, cross-validation, and error analysis. It represents the model's baseline performance without advanced preprocessing techniques.
   - **Contents**:
     - Data loading and parsing.
     - Basic text preprocessing (tokenization).
     - Binary feature extraction.
     - Training the SVM classifier.
     - Cross-validation implementation.
     - Initial error analysis using confusion matrices.

2. **Sentiment_analysis_from_tweets_after_preprocessing.ipynb**
   - **Purpose**: This file builds upon the first by introducing optimized preprocessing and feature extraction techniques aimed at improving the model's accuracy. It reflects the iterative enhancement of the model based on the insights gained from the initial error analysis.
   - **Contents**:
     - Advanced text preprocessing, including punctuation handling, lemmatization, and the inclusion of n-grams.
     - Frequency-based feature extraction.
     - Retraining the SVM classifier with the new features.
     - Cross-validation with enhanced preprocessing.
     - Detailed error analysis and performance comparison.
     - Visualization of results pre and post optimization.

## Project Structure
The project is structured into five main stages, as follows:

### 1. Simple Data Input and Pre-Processing
- **parse_data_line()**: Extracts labels and texts from the dataset.
- **pre_process()**: Tokenizes texts and conducts initial cleaning.

### 2. Simple Feature Extraction
- **to_feature_vector()**: Converts tokens to a binary feature vector and updates a global feature dictionary.

### 3. Cross-Validation on Training Data
- **cross_validate()**: Implements 10-fold cross-validation to evaluate the SVM classifier's performance on the training data.

### 4. Error Analysis
- **confusion_matrix_heatmap()**: Generates confusion matrices to identify false positives and false negatives, highlighting potential areas for model improvement.

### 5. Optimising Pre-Processing and Feature Extraction: 
Significant enhancements in preprocessing and feature extraction were implemented to improve the classifier's accuracy. Here are the key changes and their impact:

### Preprocessing Enhancements:
- **Punctuation Handling**: Originally, text processing did not separately handle punctuation, which could attach to words and affect the classifier. By separating punctuation from words, we ensured more accurate tokenization and feature extraction.
- **Lemmatization**: We introduced lemmatization to consolidate different forms of a word into a single item (lemma), reducing the feature space and focusing on the meaning rather than form of words.
- **N-grams**: Incorporation of bi-grams and tri-grams allowed the classifier to capture more contextual information compared to single-word tokens, providing insights into phrase sentiment which is often pivotal in understanding overall sentiment.

### Feature Extraction Improvements:
- **Binary to Frequency-Based Features**: Initially, features were binary (word present or not). We shifted to a frequency-based approach where the frequency of each word's occurrence in a tweet was considered, giving more weight to prevalent terms which might have a stronger sentiment indication.

### Results of Enhancements:
These preprocessing steps led to more nuanced feature sets that better represent the subtleties of language used in tweets. The changes contributed to an increase in model accuracy from 83.04% to 86.67%, reflecting a more precise understanding of tweet sentiments. The comparison of performance metrics before and after the preprocessing optimizations is shown below:

| Metric     | Before Optimization | After Optimization |
|------------|---------------------|--------------------|
| Accuracy   | 83.04%              | 86.67%             |
| Precision  | 82.86%              | 86.54%             |
| Recall     | 83.04%              | 86.67%             |
| F1 Score   | 82.88%              | 86.48%             |

These metrics demonstrate the effectiveness of our preprocessing enhancements, with particular improvements in precision and recall, crucial for a balanced sentiment analysis model.

## Results and Discussion
Initial cross-validation yielded an accuracy of 83.04%, with subsequent optimizations improving this to 86.67%. These enhancements were primarily due to more sophisticated feature extraction and data preprocessing techniques. The confusion matrix below illustrates the classifier's performance after optimizations:
![Confusion Matrix]([(https://drive.google.com/file/d/1GJlzwBHjyYv5EDqJAPcefS4wYmZeIl9b/view?usp=sharing])

## Future Improvements
Future work may explore:
- The integration of more complex NLP techniques such as sentiment-specific word embeddings.
- Expansion of the feature set to include emoji sentiment analysis.
- Application of deep learning models for potentially higher accuracy and robustness.
