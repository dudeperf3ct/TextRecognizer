# TextRecognizer

## Experiment-1

The goal of this experiment will be simple which is to solve a simplified version of line text recognition problem, a character recognizer.

The dataset we will be using for this task will be [EMNIST](https://www.nist.gov/node/1298471/emnist-dataset), which thanks [Cohen and et al](http://arxiv.org/pdf/1702.05373) it is labelled.

Here we experimented with 3 different architecture lenet, resnet and a custom CNN architecture. 

**Results**

- **Lenet**

<p>
<img src='/images/dl_project/lenet_lr.png' width="30%"/>
<img src='/images/dl_project/train_lenet.png' width="30%"/>
<img src='/images/dl_project/val_lenet.png' width="30%"/>
</p>


- **Resnet**

<p>
<img src='/images/dl_project/resnet_lr.png' width="30%"/>
<img src='/images/dl_project/train_resnet.png' width="30%"/>
<img src='/images/dl_project/val_resnet.png' width="30%"/>
</p>

- **Custom**

<p>
<img src='/images/dl_project/customCNN_lr.png' width="30%"/>
<img src='/images/dl_project/train_customCNN.png' width="30%"/>
<img src='/images/dl_project/val_customCNN.png' width="30%"/>
</p>


- **Evaluation on Test dataset**

Breakdown of classification for test dataset using above 3 architectures.

<p>
<img src='/images/dl_project/lenet_1.png' width="30%"/>
<img src='/images/dl_project/resnet_1.png' width="30%"/>
<img src='/images/dl_project/custom_1.png' width="30%"/>
</p>


<p>
<img src='/images/dl_project/lenet_2.png' width="30%"/>
<img src='/images/dl_project/resnet_2.png' width="30%"/>
<img src='/images/dl_project/custom_2.png' width="30%"/>
</p>


<p>
<img src='/images/dl_project/lenet_3.png' width="30%"/>
<img src='/images/dl_project/resnet_3.png' width="30%"/>
<img src='/images/dl_project/custom_3.png' width="30%"/>
</p>


<p>
<img src='/images/dl_project/lenet_4.png' width="30%"/>
<img src='/images/dl_project/resnet_4.png' width="30%"/>
<img src='/images/dl_project/custom_4.png' width="30%"/>
</p>

<p align="center">
<img src='/images/dl_project/lenet_sample.png' width="100%"/> 
</p>


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


## Experiment-2

Next, we will build a Line Text Recognizer. Given a image of line of words, the task will be to output what characters are present in the line.

We will use sliding window of CNN and LSTM along with [CTC loss](https://distill.pub/2017/ctc/) function.

<p align="center">
<img src='/images/dl_project/line_text.png' width="60%"/> 
</p>

For this we will use a synthetic dataset by constructing sentences using EMNIST dataset and also use [IAM dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) for training. 

We first constructed EMNIST Lines dataset. To construct this dataset we used characters from EMNIST dataset and text from brown corpus from nltk. We fixed the number of characters in each line to be 34. The new shape of image in the dataset will be (28, 28*34). The image below show some sample examples from EMNIST lines dataset.

<p align="center">
<img src='/images/dl_project/emnist_lines_sample.png' width="90%"/> 
</p>

We started with simplest model i.e. to use only CNN to predict the characters in the lines. We tried using 3 different architectures same as above lenet, resent and custom. We achieved character accuracy of 1%, 0.017% and 3.6%. 

- **Lenet CNN**

<p>
<img src='/images/dl_project/lenet_cnn.png' width="90%"/>
</p>

- **Resnet CNN**

<p>
<img src='/images/dl_project/resnet_cnn.png' width="90%"/>
</p>

- **Custom CNN**

<p>
<img src='/images/dl_project/custom_cnn.png' width="90%"/>
</p>

Next, building a complex model. We created a CNN-LSTM model with CTC loss with 3 different CNN architectures like lenet, resnet and custom as backbone. The results were remarkable. We achieved an character accuracy of 95% with lenet and 96% with custom architecture.

- **Lenet and Custom LSTM-CTC Model**

<p>
<img src='/images/dl_project/lenet_ctc_1.png' width="40%"/>
<img src='/images/dl_project/custom_ctc_1.png' width="40%"/>
</p>

<p>
<img src='/images/dl_project/lenet_ctc_2.png' width="40%"/>
<img src='/images/dl_project/custom_ctc_2.png' width="40%"/>
</p>

<p>
<img src='/images/dl_project/lenet_ctc_3.png' width="40%"/>
<img src='/images/dl_project/custom_ctc_3.png' width="40%"/>
</p>

- **Lenet LSTM-CTC Model**

<p>
<img src='/images/dl_project/lenet_ctc.png' width="90%"/>
</p>

- **Custom LSTM-CTC Model**

<p>
<img src='/images/dl_project/custom_ctc.png' width="90%"/>
</p>

Now we tried the same model with just changing the dataset. We replaced EMNIST Lines with IAM Lines dataset.

<p align="center">
<img src='/images/dl_project/iam_lines_sample.png' width="90%"/> 
</p>

And the results.

- **Lenet and Custom LSTM-CTC Model**

<p>
<img src='/images/dl_project/lenet_iam_1.png' width="40%"/>
<img src='/images/dl_project/custom_iam_1.png' width="40%"/>
</p>

<p>
<img src='/images/dl_project/lenet_iam_2.png' width="40%"/>
<img src='/images/dl_project/custom_iam_2.png' width="40%"/>
</p>

- **Lenet LSTM-CTC Model**

<p>
<img src='/images/dl_project/lenet_iam.png' width="90%"/>
</p>

- **Custom LSTM-CTC Model**

<p>
<img src='/images/dl_project/custom_iam.png' width="90%"/>
</p>


**Learnings**

- Switching datasets worked but still requires a lot of time to train for further fine prediction i.e train more.
- LSTM involves a lot many experiments use bidirectional or not, use gru or lstm. Trying different combinations might help get even better results for each CNN architecture.
- Further, we can make use of attention-based model and use language models which will make model more robust.
- Using beam search decoding for CTC Models


## Experiment-3

Almost done! We have completed Line Text predictor. Now comes the part of implementing Line Detector. For this, we will use IAM dataset again but paragraph dataset. Here is a sample image from paragraph dataset.

<p>
<img src='/images/dl_project/sample_1.jpg' width="40%"/>
<img src='/images/dl_project/sample_2.jpg' width="40%"/>
</p>

The objective in this experiment is to design a line detector. Given a paragraph image the model must be able to detect each line. What do you mean by detect? We will preprocess the paragraph dataset such that each pixel corresponds to either of the 3 classes i.e. 0 if it belongs to background, 1 if it belongs to odd numbered line and 2 if it belongs to even numbered line. Wait, why do you need 3 classes, when 2 are sufficient? The image below explains why we need 3 classes instead of 2?

With 2 classes : 0 for background and 1 for pixels on line.

<p>
<img src='/images/dl_project/only_2.png' width="90%"/>
</p>  
 
With 3 classes : 0 for background, 1 for odd numbered-line and 2 for even numbered-line.
 
<p> 
<img src='/images/dl_project/only_3.png' width="90%"/>
</p>

Here is how our dataset for line detection will look like after preprocessing.

<p>
<img src='/images/dl_project/para_ex1.png' width="90%"/>
</p>

<p>
<img src='/images/dl_project/para_ex2.png' width="90%"/>
</p>

Here is a sample after apply data augmentation.

<p>
<img src='/images/dl_project/para_aug_ex1.png' width="90%"/>
</p>


Now that we have dataset, images with paragraph of size (256, 256) and ground truths of size (256, 256, 3) we use full convolution neural networks to give output of size (256, 256, 3) for an input of (256, 256). We use 3 architectures, lenet-FCN (converted to FCNN), resnet-FCN and custom-FCN.

Results are bit embarassing.

- **Lenet-FCN**

<p>
<img src='/images/dl_project/lenet_iam_para.png' width="90%"/>
</p>


- **Resnet-FCN**

<p>
<img src='/images/dl_project/resnet_iam_para.png' width="90%"/>
</p>


- **Custom-FCN**

<p>
<img src='/images/dl_project/custom_iam_para.png' width="90%"/>
</p>


**Learnings**

- Investigate as to why model is not performing well in segmenting. Having a good line segmentor is critical for our OCR pipeline.



## Experiment-4

Finally, all pieces from above experiments come together. To recap, we have a Line Predictor Model from experiment-2 which takes in input images of lines and predicts the characters in the line. And we have a Line Detector Model from experiment-3 which segments paragraphs into line regions.

Do you see the whole picture coming together? No?

<p align="center">
<img src='/images/dl_project/computer-vision.jpg' width="60%"/>
</p>

1. Given an image like the one above, we want a model that returns all the text in the image.
2. First step, we would use Line Detector Model. This model will segment image into lines.
3. We will extract crops of the image corresponding to the line regions obtained from above line and pass it to Line Predictor Model which will predict what characters are present in the line region.
4. Sure enough if both the models are well trained, we will get excellent results!


## Experiment-5

Now that we have full end-to-end model, we can run the same model on a web server or create an android app.