Experiment-2
==============================

In this experiment, we will build a Line Text Recognizer. Given a image of line of words, the task will be to output what characters are present in the line.

We will use sliding window of CNN and LSTM along with [CTC loss](https://distill.pub/2017/ctc/) function.

For this we will use a synthetic dataset by constructing sentences using EMNIST dataset and also use [IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) for training. 

We first constructed EMNIST Lines dataset. To construct this dataset we used characters from EMNIST dataset and text from brown corpus from nltk. We fixed the number of characters in each line to be 34. The new shape of image in the dataset will be (28, 28*34). 

We started with simplest model i.e. to use only CNN to predict the characters in the lines. We tried using 3 different architectures same as above lenet, resent and custom. We achieved character accuracy of 1%, 0.017% and 3.6%. 

Next, building a complex model. We created a CNN-LSTM model with CTC loss with 3 different CNN architectures like lenet, resnet and custom as backbone. The results were remarkable. We achieved an character accuracy of 95% with lenet and 96% with custom architecture.


**Learnings**

- Switching datasets worked but still requires a lot of time to train for further fine prediction i.e train more.
- LSTM involves a lot many experiments use bidirectional or not, use gru or lstm. Trying different combinations might help get even better results for each CNN architecture.
- Further, we can make use of attention-based model and use language models which will make model more robust.
- Using beam search decoding for CTC Models


A short description of the project.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    |
    |
    └── src                 <- Source code for use in this project
        ├── data
        │   ├── dataset.py
        │   ├── emnist_dataset.py  <- Scripts to download or generate data
        │   ├── emnist_lines.py
        │   ├── iam_lines.py
        |
        ├── __init__.py            <- Makes src a Python module
        |
        ├── models
        │   ├── base_model.py
        │   ├── line_model_ctc.py
        │   ├── line_model.py
        |
        ├── networks
        │   ├── cnn_lstm_ctc.py
        │   ├── ctc.py
        │   ├── custom_cnn.py
        │   ├── custom.py
        │   ├── lenet_cnn.py
        │   ├── lenet.py
        │   ├── resnet_cnn.py
        │   ├── resnet.py
        │   └── sliding.py
        |
        ├── tests                    <- Scripts to use trained models to make predictions
        │   ├── support
        │   ├── create_emnist_lines.py
        │   └── test_character_predictor.py  
        ├── training                        <- Scripts to train models
        │   ├── line_predictor.py
        │   ├── clr_callback.py
        │   ├── lr_find.py
        │   ├── train_model.py                 
        │   └── util.py
        ├── util.py
        └── visualization       <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
