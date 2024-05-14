# ML_French_Blancpain
Group Members: Yue Wu, Zayna Chanel

## Video
 **[Check our video here](https://drive.google.com/input url)**


## Repository Index
- [French_difficulty_original.ipynb](/French_difficulty_original.ipynb): The python notebook for the model training with original data
- [Image](/Image): The folder of essential images in the notebook
- [streamlit](/streamlit): The folder of essential images in the notebook
- [UI_French.py](/UI_French.py): The streamlit UI 
- [data_augmentation.ipynb](/data_augmentation.ipynb): The code for data augmentation
- [back_translation.csv](/back_translation.csv): The augmented data using back translation
- [nlpaug.csv](/nlpaug.csv): The augmented data using nlp
- [unlabelled_test_data.csv](/unlabelled_test_data.csv): The unlablled data for prediction (provided by Kaggle)
- [training_data.csv](/training_data.csv): The original data for training (provided by Kaggle)
- [sample_submission.csv](/sample_submission.csv): The sample submission (provided by Kaggle)
- [README.md](/README.md): The README document

# 1.Model preparation
## 1.1 Dataset split
An essential step in preparing our dataset for model training and evaluation is to divide it into training and validation sets.

**Implementation Details:**
To split our dataset, we use the train_test_split function from the sklearn.model_selection module. We specify a test_size of 0.1, which allocates 10% of the dataset to the validation set and the remaining 90% to the training set.
**Output Variables**:
_train_texts_: Text data for training the model.
_val_texts_: Text data reserved for validating the model's performance.
_train_labels_: Difficulty labels corresponding to the training text data.
_val_labels_: Difficulty labels corresponding to the validation text data.
This split ensures that we have a robust dataset for training while also setting aside a representative portion of data for performance evaluation, crucial for fine-tuning our model parameters.

## 1.2 Data preparation
We choose to use the Bidirectional Encoder Representations from Transformers (Bert), a state-of-the-art pre-trained language model based on the Transformer architecture. Bert is highly regarded for its deep understanding of language context and its ability to perform various natural language processing (NLP) tasks. It learns language representations on a large-scale text corpus. Its strengths include powerful pre-trained language capabilities, bi-directional encoding, and transfer learning potential, which make it ideal for extracting textual features.
We implemented the text feature extraction process through a function named 'bert_feature(data, **kwargs)'. This function is designed to process a list of text data, performing tokenization and encoding to transform texts into a format suitable for further treatment. 
Before training our models, it's crucial to scale the feature data.We utilize the StandardScaler from sklearn.preprocessing to scale our feature vectors. 
For our classification task, it is necessary to convert categorical labels into a numerical format that our models can work with. We define a mapping from the categorical difficulty levels (A1, A2, B1, B2, C1, C2) to integers (0, 1, 2, 3, 4, 5). This is done to facilitate the model's ability to perform mathematical operations on the output predictions.

## 1.2 Classification and model evaluation 
After preparing our data, we proceed to evaluate various classification models to determine their effectiveness in predicting sentence difficulty. Below is a summary of the performance metrics for each model we tested.

**Evaluation Metrics:**
_Accuracy_: Proportion of total predictions that were correct.
_Precision_: Proportion of positive identifications that were actually correct.
_Recall_: Proportion of actual positives that were identified correctly.
_F1-Score_: Harmonic mean of precision and recall, providing a single metric that balances both.

**Model Performance Summary:**
We tested a variety of machine learning models to find the best performer for our  task. The models evaluated include **Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Extra Trees, Support Vector Machine,** and several **boosting algorithms (XGBoost, LightGBM, CatBoost)**, as well as a **Multi-Layer Perceptron (MLP)**. 

To enhance the performance of our RandomForest and ExtraTrees classifiers, we employed GridSearchCV from sklearn.model_selection as hyper-parameter optimization to find the best solution. This tool allows us to systematically explore multiple combinations of parameter tunes, and determine which parameters yield the best model performance.
![hyper-parameter](Image/hypertext_original.png)

In our evaluation of various machine learning models, the Support Vector Machine (SVM) exhibited the highest overall accuracy and F1-score, highlighting its superior performance for our text difficulty prediction task. This model effectively managed to balance precision and recall, making it highly suitable for our application.

Here's a quick summary of how the SVM stands out and some notes on other notable models:

Support Vector Machine (SVM): SVM achieved the highest F1-score among all the models tested, demonstrating its efficacy in handling the complex patterns in our feature data derived from the CamemBERT model. Its robust performance in both precision and recall makes it particularly valuable for predicting sentence difficulty where maintaining a balance between false positives and false negatives is crucial.

Boosting Algorithms: While models like XGBoost and LightGBM also showed competitive performances, they did not surpass the SVM in terms of overall metrics. However, their ability to adaptively enhance their performance through iterations still makes them strong contenders for further tuning.

CatBoost: Despite earlier claims, while CatBoost performed well, especially in terms of F1-score, it did not achieve the highest score across all metrics. This indicates a strong performance but suggests there may be room for optimization or it may be more suited to specific types of classification tasks within our dataset.

Random Forest (improved): This model showed a marked improvement over the basic Random Forest, indicating that enhancements such as parameter tuning can significantly boost performance.

![Accuracy_table](/Image/accuracy_table_original.png)


![Confusion Matrix](/Image/CM_SVC_original.png)

Then we use the whole dataset to retrain the model and use the retrained model to predict the difficulty level in the unlabelled test dataset.
