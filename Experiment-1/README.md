Experiment-1
==============================

The goal of this experiment will be simple which is to solve a simplified version of line text recognition problem, a character recognizer.

The dataset we will be using for this task will be [EMNIST](https://www.nist.gov/node/1298471/emnist-dataset), which thanks [Cohen and et al](http://arxiv.org/pdf/1702.05373) it is labelled.

Here we experimented with 3 different architecture lenet, resnet and a custom CNN architecture. 

Refer the [blog](https://dudeperf3ct.github.io/project/2019/05/17/Fun-in-Deep-Learning-Project/#experiment-1) for further details.

**Learnings**

- Initially we trained all models with a constant learning rate.
- Instead of using constant learning rate, we implemented cyclic learning rate and learning rate finder which provided a great boost in terms of both speed and accuracy for performing various experiments.
- Transfer learning with resnet-18 performed poorly.
- From above results of test evaluation, we can see that model performs poorly on specific characters as there can be confusion due to similarity like digit 1 and letter l, digit 0 and letter o or O, digit 5 and letter s or S or digit 9 and letter q or Q.
- Accuracies on train dataset are 78% on lenet, 83% on resnet and 84% on custom.
- Accuracies on val dataset are 80% on lenet, 81% on resnet and 82% on custom.
- Accuracies on test dataset are 62% on lenet, 36% on resnet and 66% on custom.
- Custom architecture performs well but resnet perform poorly (Why?)
- There is a lot of gap in train-val and test even when val distribution is same as test distribution i.e. val set is taken from 10% of test set.
- Look for new ways to increase accuracy



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
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    |
    |
    ├── src                <- Source code for use in this project.
        ├── data
        │   ├── dataset.py
        │   ├── emnist_dataset.py  <- Scripts to download or generate data
        |
        ├── __init__.py            <- Makes src a Python module
        |
        ├── models
        │   ├── base_model.py
        │   ├── character_model.py
        |
        ├── networks
        │   ├── custom.py
        │   ├── lenet.py
        │   └── resnet.py
        |
        ├── tests                    <- Scripts to use trained models to make predictions
        │   ├── support
        │   │   ├── create_emnist_support_files.py
        │   │   └── emnist
        │   │       ├── 3.png
        │   │       ├── 8.png
        │   │       ├── a.png
        │   │       ├── e.png
        │   │       └── U.png
        │   └── test_character_predictor.py  
        ├── training                        <- Scripts to train models
        │   ├── character_predictor.py
        │   ├── clr_callback.py
        │   ├── lr_find.py
        │   ├── train_model.py                 
        │   └── util.py
        ├── util.py
        └── visualization       <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
