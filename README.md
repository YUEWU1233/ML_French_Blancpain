# ML_French_Blancpain
Group Members: Yue Wu, Zayna Chanel

## Video
 **[Check our video here](https://drive.google.com/input url)**


## Repository Index
- [French_difficulty_original.ipynb](/French_difficulty_original.ipynb): The python notebook for the model training with original data
- [Image](/Image): The folder of essential images in the notebook
- [UI_French.py](/UI_French.py): The streamlit UI 
- [data_augmentation.ipynb](/data_augmentation.ipynb): The code for data augmentation
- [back_translation.csv](/back_translation.csv): The augmented data using back translation
- [nlpaug.csv](/nlpaug.csv): The augmented data using nlp
- [unlabelled_test_data.csv](/unlabelled_test_data.csv): The unlablled data for prediction (provided by Kaggle)
- [training_data.csv](/training_data.csv): The original data for training (provided by Kaggle)
- [sample_submission.csv](/sample_submission.csv): The sample submission (provided by Kaggle)
- [README.md](/README.md): The README document

# 1.Model preparation
## 1.1 Tokenization and Feature extraction
We chose to use the Bidirectional Encoder Representations from Transformers (Bert), a state-of-the-art pre-trained language model based on the Transformer architecture. Bert is highly regarded for its deep understanding of language context and its ability to perform various natural language processing (NLP) tasks. It learns language representations on a large-scale text corpus. Its strengths include powerful pre-trained language capabilities, bi-directional encoding, and transfer learning potential, which make it ideal for extracting textual features.
We implemented the text feature extraction process through a function named bert_feature(data, **kwargs). This function is designed to process a list of text data, performing tokenization and encoding to transform texts into a format suitable for the Bert model. 

## 1.2 Classification
Firstly, we devide our dataset for model training and evaluation into training and validation sets at a 90% and 10% ratio. 
Then we use different classification model to train our scaled extracted features, and evaluate their accuracy.
![Accuracy_table](/Image/accuracy_table_original.png)


![Confusion Matrix](/Image/CM_SVC_original.png)
