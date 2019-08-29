Experiment-4
==============================
Finally, all pieces from above experiments come together. To recap, we have a Line Predictor Model from experiment-2 which takes in input images of lines and predicts the characters in the line. And we have a Line Detector Model from experiment-3 which segments paragraphs into line regions.

Do you see the whole picture coming together? No?

1. Given an image like the one above, we want a model that returns all the text in the image.
2. First step, we would use Line Detector Model. This model will segment image into lines.
3. We will extract crops of the image corresponding to the line regions obtained from above line and pass it to Line Predictor Model which will predict what characters are present in the line region.
4. Sure enough if both the models are well trained, we will get excellent results!


Refer the [blog](https://dudeperf3ct.github.io/project/2019/05/17/Fun-in-Deep-Learning-Project/#experiment-4) for further details.

**Learnings**

- Investigate as to why model is not performing well in segmenting. Having a good line segmentor is critical for our OCR pipeline.



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
        │   ├── emnist_dataset.py
        │   ├── emnist_lines.py
        │   ├── iam_dataset.py   <- Scripts to download or generate data
        │   ├── iam_lines.py
        │   ├── iam_paragraphs.py
        |
        ├── __init__.py            <- Makes src a Python module
        |
        ├── models
        │   ├── base_model.py
        │   ├── line_detect_model.py
        │   ├── line_detect_ctc.py
        |
        ├── networks
        │   ├── cnn_lstm_ctc.py
        │   ├── ctc.py
        │   ├── custom_cnn.py
        │   ├── custom_fcn.py
        │   ├── custom.py
        │   ├── fcn.py
        │   ├── __init__.py
        │   ├── lenet_cnn.py
        │   ├── lenet_fcn.py
        │   ├── lenet.py
        │   ├── resnet_cnn.py
        │   ├── resnet.py
        │   └── sliding.py
        |
        ├── paragraph_text_recognizer.py
        ├── util.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>