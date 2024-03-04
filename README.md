# HATE SPEECH DETECTION PROJECT

## Overview

This project focuses on the development of a hate speech detection system using machine learning techniques. The implementation includes importing necessary modules, describing the dataset, creating a cleaning function, implementing text processing techniques such as stopwords removal and stemming, and using the CountVectorizer and DecisionTree for model development.

## Table of Contents

Introduction\
Installation\
Data\
Cleaning Funciton\
Model Developmnt\
Confusion Matrix and Accuracy\
Results\
Contributing\
License

## Introduction

Hate speech is a significant issue in online communication, and this project aims to develop a machine learning model for detecting hate speech. The project utilizes natural language processing (NLP) techniques and machine learning algorithms, such as DecisionTree, for hate speech classification.


## Installation
To run this project locally, follow these steps:

1. Clone the repository:
```python 
git clone https://github.com/your-username/hate-speech-detection.git
```
2. Install the required dependencies:
```python
pip install -r requirements.txt
```
## Data

The hate speech detection model was trained on a Twitter dataset, which is included in this repository. The dataset has the following structure:

- **File Format:** CSV
- **Number of Entries:** 24,783
- **Columns:**
  1. `Unnamed: 0`: Index column
  2. `count`: Count information
  3. `hate_speech`: Count of hate speech occurrences
  4. `offensive_language`: Count of offensive language occurrences
  5. `neither`: Count of neutral occurrences
  6. `class`: Class label (0 for Hate Speech, 1 for Offensive Speech, 2 for Neither)
  7. `tweet`: Text content of the tweet

### Label Encoding

A label encoding step was performed during data preprocessing to create a new column `labels`, representing the following categories:

- 0: "Hate Speech Detected"
- 1: "Offensive Speech Detected"
- 2: "Neither Hate Nor Offense"

This label encoding facilitates the training and evaluation of the hate speech detection model.

The dataset is available in the file `twitter_data.csv` within this repository.

## Cleaning Function

A cleaning function was applied to the text data in the Twitter dataset to prepare it for hate speech detection. The function, `clean_data`, performs the following cleaning steps:

1. **Convert to Lowercase:**
   - All text was converted to lowercase to ensure uniformity.

2. **Remove URL and Web Links:**
   - Regular expressions were used to remove URLs and web links from the text.

3. **Remove Squared Braces:**
   - Squared braces and their contents were removed from the text.

4. **Remove Angle Brackets:**
   - Angle brackets and their contents were removed from the text.

5. **Remove Punctuation:**
   - All punctuation marks were removed from the text.

6. **Remove Newline Characters:**
   - Newline characters were replaced with empty strings.

7. **Remove Words Containing Digits:**
   - Words containing digits were removed from the text.

8. **Stopwords Removal:**
   - Stopwords removal was applied to further clean the text.
      - Stopwords are common words (e.g., "the", "is", "and") that are often excluded from text data as they do not carry significant meaning.

The cleaned text was then updated in the 'tweet' column of the dataset.

### Example Usage

```python
# Import the required libraries
import re
import string

# Define the cleaning function
def clean_data(text):
    # ... (as mentioned above)

# Apply the cleaning function to the 'tweet' column
df['tweet'] = df['tweet'].apply(clean_data)
print(df.head())
```

## Model Development

The hate speech detection model was developed using a Decision Tree classifier. The following steps outline the model development process:

1. **Text Vectorization:**
   - The 'tweet' column containing cleaned text data was converted into numerical vectors using the `CountVectorizer`.
   - `CountVectorizer` converts a collection of text documents to a matrix of token counts.

   ```python
   import numpy as np
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier

   # Extract features (x) and labels (y) from the dataset
   x = np.array(df['tweet'])
   y = np.array(df['labels'])

   # Initialize CountVectorizer
   cv = CountVectorizer()

   # Convert text data to numerical vectors
   x = cv.fit_transform(x)

   # Split the dataset into training and testing sets
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

2. Decision Tree Model Training:

A Decision Tree classifier (DecisionTreeClassifier) was chosen for its simplicity and interpretability.
The model was trained on the training dataset.

```python
# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the model on the training data
clf.fit(x_train, y_train)
```

3. Model Evaluation:

The trained model was evaluated on the testing dataset, and predictions (y_pred) were obtained.

```python
# Make predictions on the test set
y_pred = clf.predict(x_test)
```

## Confusion Matrix and Accuracy

To assess the performance of the hate speech detection model, a confusion matrix and accuracy were computed using the test dataset. The following steps were taken:

1. **Compute Confusion Matrix:**
   - The `confusion_matrix` function from `sklearn.metrics` was used to generate a confusion matrix.
   - The confusion matrix compares the predicted labels (`y_pred`) with the true labels (`y_test`).

   ```python
   from sklearn.metrics import confusion_matrix

   # Compute the confusion matrix
   cm = confusion_matrix(y_test, y_pred)

   
2. Visualize Confusion Matrix:

The confusion matrix was visualized using a heatmap for better interpretation.
The seaborn library was utilized to create the heatmap.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the confusion matrix using a heatmap
sns.heatmap(cm, annot=True, fmt=".1f")
plt.show()
```
*The heatmap provides a graphical representation of the true positive, true negative, false positive, and false negative predictions.*

## Results

### Example Predictions

You can use the trained hate speech detection model to make predictions on new text data. Here's an example of how to use the model for predictions:

```python
# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Assuming 'cv' is the CountVectorizer object and 'clf' is the trained Decision Tree classifier
# ... (as mentioned above)

# Example Text Data for Testing
text_data = "I’m not stopping by my place but now I am on the galaxy of my love with my love of my baby girl..."
text_data1 = "Shut up you Bitch"

# Convert the example text data to numerical vectors
df = cv.transform([text_data, text_data1]).toarray()

# Make predictions using the trained model
predictions = clf.predict(df)

# Display the predictions
print(predictions)
```

## License

This project is licensed under the MIT License.
