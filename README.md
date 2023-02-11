# BERT Fine-tuning for Acceptability Judgment Task

This repository contains a Jupyter Notebook implementation of fine-tuning the pre-trained BERT model for Acceptability Judgment task. The model is fine-tuned on a custom dataset using the transfer learning approach, which allows us to leverage the pre-trained knowledge learned by BERT on large amounts of data and apply it to the task of Acceptability Judgment.

## Requirements
To run the code in this repository, you will need to have the following packages installed:
* PyTorch
* Transformers
* Pandas
* Numpy

## Data
The dataset used for fine-tuning is expected to be in the format of a .tsv file, with each row containing the text and the corresponding label.

Before fine-tuning the pre-trained BERT model, the data was pre-processed in the following way:
* The special tokens [CLS] and [SEP] were added to the beginning and end of each text, respectively.
* The text was tokenized using the BERT tokenizer.
* The tokenized text was converted to token IDs.
* The token IDs were padded to make them of equal length.
* Attention masks were created to ensure that the model only predicts on the non-zero IDs.

This pre-processing step is crucial for fine-tuning the BERT model as it ensures that the input to the model is in the correct format.

## Usage
To run the Jupyter Notebook, simply open the FineTuning_BERT.ipynb and follow the instructions in the notebook. The notebook contains the following sections:
1. Load the data
2. Pre-processing
3. Fine-tuning
4. Evaluation
5. Inference on new data

## Evaluation Metric
To evaluate the performance of the fine-tuned BERT model, Matthews Correlation Coefficient (MCC) was used. MCC is a commonly used evaluation metric in binary classification problems, as it takes into account both the true positive and true negative rates, while also considering the false positive and false negative rates. MCC values range from -1 to 1, with 1 indicating a perfect prediction and -1 indicating a completely incorrect prediction.

## References
This project was inspired by the following resources:
* Transformers for Natural Language Processing by Denis Rothman: A comprehensive guide to transformers and their applications in NLP.

I would like to express my gratitude to the authors for their invaluable contributions to the field of NLP and for making this knowledge accessible to everyone.

## Acknowledgements
I would like to acknowledge the HuggingFace team for providing the pre-trained BERT model and the Transformers library, which made this project possible. I would also like to thank the developers of PyTorch for providing a flexible and user-friendly deep learning framework.
