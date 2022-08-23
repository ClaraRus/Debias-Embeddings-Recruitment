# Debiasing word embeddings with respect to gender for the recruitment industry
The recruitment industry relies more and more on automation for processing, searching, and matching job vacancies to job seekers. However, automation of the recruitment process can lead to discriminatory results with respect to certain groups, based on gender, ethnicity or age. One reason for unfair behavior could be the input data. If the data from which the model learns is biased the output could also be biased. This research presented three methods for identifying the presence of bias in recruitment texts: the Word Embedding Association Test, predicting the gender from the representations, and the Salary Association Test. One way to learn useful debiased representations is through adversarial learning. An encoder is trained together with a classifier and an adversary to ensure both the usefulness and the fairness of the embeddings. The adversary tries to predict the sensitive variable from the produced representation. The goal is to learn fair representations that are also useful for the prediction task. 

This repository provides code for running the adversarial debiasing method on the [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult). Moreover, it provides code samples for identifying the presence of bias in the data using the above mentioned methods.

## Prerequisites

```
python 3.6
pytorch 1.4.0
pandas 1.1.5
numpy 1.17.0
```
## Setup
Clone the repository:
```
git clone https://github.com/ClaraRus/Debias-Embeddings-Recruitment.git
cd Debias-Embeddings-Recruitment
```
Create conda environment and install the required python packages:
```
conda create -n adversarial_debiasing python=3.6
conda activate adversarial_debiasing
pip install -r requirements.txt
```
## Training Arguments
In the ```configs``` folder you can find examples of config files to run the training of the adversarial debiasing method.
```
{
     "batch_size": 100,
     "epochs": 5,
     "n_save": 1,
     "logits": false,
     "alternate_batch": false,
     "penalize": false,
     "train_adv": true,
     "train_cla": true,
     "train_dec": false,
     "fair_loss": false,
     "lam": [1, 1, 0],
     "lr": 1e-5,
     "decay":1e-4,
     "EXP": 1.1,
     "dataset": "adult",
     "root_path": "../datasets/adult/"
}
```
* batch_size - The batch size used during training and validation
* epochs - Number of epochs to run the training
* n_save - How often to save the model
* logits - Train the model with weighted loss for the positive class
* alternate_batch - Train the model by alternating the update of the components
* penalize - Penalize the adversary if it is too strong, otherwise penalize the classifier if the adversary is too weak
* train_adv - Train the adversary component
* train_cla - Train the classifier component
* train_dec - Train the decoder component
* fair_loss - Copmute the loss using the fairness metrics. Available values: "parity" - to use statistical parity
* lam - Hyperparameters for computing the final loss
* lr - Learning rate
* EXP - Experiment Number
* dataset - Name of the dataset to train the model on. Available values: "adult" to train on the Adult dataset.
* root_path - Path to the folder containing the dataset.

## Run Training
The code runs on the Adult dataset. You can implement your own dataset class in ```dataset.py```.

Run the following command to download the Adult dataset:
```
mkdir datasets
cd datasets
mkdir adult
cd adult
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
cd ..
cd ..
```

To start training the model run the following command:
```
cd adversarial_debiasing
python run_training.py --path_configs ../configs --experiments 1 --runs 1
```
* path_configs - Path to folder containing the configs files
* experiments - Number of experiments to run
* runs - Number of runs per experiment
                   

## Run Evaluation
During training evaluation is performed on the final model. The notebook ```Compare Results.ipynb``` contains example of how to evaluate the models and compare the results of the runs.

## Output Files
Output files are saved in the folder ```models/dataset_name/experiment_name/run_number```. For example the training code above will create the following output folder: ```models/adult/FairGAN-cla-adv-EXP-1.1/0/```. 

For each run the following files are saved during training:
* config.json- the config file used to run the training
* train_log.txt - training logs
* val_log.txt - validation logs
* model.pt - final model
* model-n.pt - model saved at epoch n
* eval - evaluation results of final model
* eval-epoch<n> - evaluation results of model at epoch n
* eval_class.csv - performance of the classifier on each class
* fairness_metrics.csv - fairness of the classifier on each class
* ROC_adv.png - Receiver operating characteristic curve of the adversary on predicting the gender
* ROC_class.png - Receiver operating characteristic curve of the classifier for each class

## Identify Bias
This research presented three methods for identifying the presence of bias in recruitment texts: the Word Embedding Association Test, predicting the gender from the representations, and the Salary Association Test. Examples of notebooks can be found in ```identify_bias```. 
* **Word Embedding Association Test**: The notebook ```WEAT.ipynb``` shows examples of how to run the WEAT on pre-trained Word2Vec and GloVe models. The following test-cases are present in this repo: the original tes-cases ```weat_test_cases.py```, the English recruitment test-cases ```weat_test_cases_recruitment_eng.py```, and the Dutch recruitment test-cases ```weat_test_cases_recruitment_dutch.py```.
* **Predicting the gender**: The notebook ```Predict gender.ipynb``` shows an example on the Adult dataset of how to run a classifier to predict the gender from the data. 
* **Salary Associaiont Test**: no code available.


## Acknowledgements
* Code used for the adversarial debiasing method was based on the following repo: https://github.com/oozdenizci/AdversarialEEGDecoding
* Code used for the WEAT was based on the following repo: https://github.com/chadaeun/weat_replication/tree/0753713a47333827ef9f653d85e08740834ef698 
    
