Project Overview
The project applies a Naive Bayes classifier, a popular choice for text classification tasks, especially spam detection. The model predicts the probability that a message is spam or not, using features derived from the text itself.

Key performance metrics include:
Accuracy
Precision
Recall
F1 Score

Dataset
The dataset used in this project consists of labeled text messages, with each message marked as either spam (1) or not spam (0).

Column	Description
label	1 if the message is spam, 0 otherwise
message	Text of the message
Note: Ensure the dataset is in the appropriate format, such as spam.csv.

Requirements
Python 3.7+
pandas
scikit-learn
nltk
joblib (for saving the model)

Usage
1. Preprocess the Data
Ensure stop words are removed and text is tokenized before training:

python
Copy code
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# Additional data preprocessing steps as required

2. Train the Model
Use the train_model.py script to train the Naive Bayes model:

python
Copy code
python src/train_model.py
3. Save the Model
The trained model is saved in the models/ directory as naive_bayes_model.joblib.

4. Evaluate the Model
Run the evaluate_model.py script to view accuracy, precision, recall, and F1 score:

python
Copy code
python src/evaluate_model.py
5. Predict New Messages
Use the predict.py script to classify new messages as spam or not spam:


Model Performance
The model's performance metrics on the test set are as follows:

Metric	Score
Accuracy	98.48%
Precision: 97%
Recall: 91%
F1 Score: 94%


Results
The Naive Bayes model is highly effective for binary text classification tasks like spam detection, thanks to its simplicity and efficiency with text data.

Sample Predictions
Message Text	Predicted Label
"Click the link to get a $10 reward today!!!"	Spam
"Are you done with the Data Structures assignment?"	Not Spam

Future Work
Future improvements to this project could include:
Experimenting with different vectorization techniques (TF-IDF).
Using an ensemble method to combine Naive Bayes with other classifiers for better accuracy.
