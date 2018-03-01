# **Traffic Sign Recognition** 

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results


[//]: # (Image References)

## Rubric Points
What is included in this project (git repo) are
   + jupyter iPython notebook file (Traffic_Sign_Classifier.ipynb)
   + HTML output of the results of executing the notebook (checked in as Traffic_Sign_Classifier.html)
   + This markdown sumhttps://github.com/cwcentral/CarND-Traffic-Sign-Classifier-Project/edit/home/README.mdmary

### README

### 1. Project Overview

Here is a link to the original [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Here is a link to my project code and results [project code](https://github.com/cwcentral/CarND-Traffic-Sign-Classifier-Project/blob/home/Traffic_Sign_Classifier.ipynb)

The reference links to get the test data used by this project are located [here](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/614d4728-0fad-4c9d-a6c3-23227aef8f66/project). I used the dataset in files called:
    
    + train.p
    + test.p
    + valid.p

This project and it's configuration and results can be [found here](https://github.com/cwcentral/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

Note: this project was originally developed on a NON-Nvidia computer, hence run under CPU mode.

### 2. Data Set Summary

The traffic signs data set contain the following indexes:
   * 'features' (image data in vectors of size 4, aka 4D arrays)
   * 'labels'   (based on signnames.csv classifications IDs)
   * 'sizes'    (cv like width & height)   
   * 'coords'   (tuples representing the UL/LR bounding box of the image)
   
The pandas library is used to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 3. Exploratory visualization of the dataset.

German road sign dataset:
![](markdown_data/dataset.png?raw=true)

The dataset consists of low resolution images of german street signs. These images of each classification are fairly equal (look similar) and thus we expect some accuracy issues. This was proven as desribed later, so I needed to perform some image enhancement to get more accurate representation of the signs.

![](markdown_data/exploratory_visualization.png?raw=true)
exploratory_visualization.png displays a sample of the images. I ran a histogram graph to show the number of samples spread among the classes in the dataset to get an idea of distribution. Since the dataset gven was in RGB-color, I converted to grayscale to speed up the CNN processing speed. In order to emphasize shape, edges, and patterns that deterimine CNN effectivness, I increase the contrast and normalize the images.

![](markdown_data/norm_images.png?raw=true)

Secondly, if one observes the distribution of the datasets, we do not see a even distribution of signs which could make our model more sensitive to the learning rate (and outliers) as well as take longer to achieve a reasonable accuracy.

![](markdown_data/dataset.png?raw=true)


### 4. Design and Test a Model Architecture

First, we need to preprocess the data to give the model a nice distributed sampling to work on.

I performed these operations:

a. convert existing data (preprocess)
   * convert to grayscale, add contrast, and normalize the pixels
b. Since the training dataset is not evenly random, we augmented by adding samples upto 400 to those 
    classes with little samples. Sampling included rotating, adding noise and blurring images.
c. sample shuffling: we need to make each set random as possible,

Here's what the augmented dataset looks like:
![](markdown_data/preprocess.png?raw=true)

Now we take the datasets and construct our CNN model. I used the standard LeNet architecture in Tensorflow. From initial run against the samples we found the model was constantly overfitting, so I added some dropout to the final output. I found around the 80% keep probablity was sufficient.

My final model consisted of the following layers:

|:---------------------:|:---------------------------------------------:| 
| Layer            		|     Description	        			         		| 
|:---------------------:|:---------------------------------------------:| 
| Convolution      		| 32x32x1 gray image, output 28x28x6            |
| RELU                	|                                               |
| Max pooling        	| Input = 28x28x6. Output = 14x14x6     .	     	|
|:---------------------:|:---------------------------------------------:| 
|						      |										            		|
| Convolution        	| Convolutional Input =14x14x6 Output =10x10x16.|
| RELU				    	|										            		|
| Max pooling	      	| Input = 10x10x16. Output = 5x5x16.		   	|
| flatten       	      | Input = 5x5x16. Output = 400.                 |
|:---------------------:|:---------------------------------------------:| 
|					      	|											            	|
| Fully connected		   | Input = 400. Output = 120.                    |
| RELU                  |                                               |
| Dropout               |                                               |
|:---------------------:|:---------------------------------------------:| 
|						      |									            			|
| Fully Connected       | Input = 120. Output = 84.                     |
| RELU                  |                                               |
| Dropout               |                                               |
|:---------------------:|:---------------------------------------------:| 
|						      |					            							|
| Fully Connected       | Input = 84. Output = n_classes.               |
| One Hot               |                                               |
| Softmax               |                                               |
|:---------------------:|:---------------------------------------------:| 

To train the model, I used the follwoing hyper parameters:

+ Number of epochs = 30
+ batch size = 96
+ learning rate = 0.002
+ dropout keep_prob = 0.8 (80%)
+ (arg for truncation) mu = 0
+ (arg for truncation) sigma = 0.1
    
As per standard LeNet architecture, I also added the AdamOptimizer.

The original LeNet design as from text produced a validation accuracy less than equal to 88% and a training accuracy of less than equal to 94%. That indicated the architecture was overfitting. Thus I had to solve this problem. Without changes to the inputs, we looked at the architecture hyperparameters:
    * Increasing the epoch to 64 had minimal effect (did hit validation accuracy 93%, but rarely)
    * Increasing the batch size made it vary more often +- 5%, diverging accruacy torwards the end of a run
    * Decreasing the batchsize made it vary less, but not reach the 93% requirement.
    * Increase learning rate did achieve iterations above 93%, but momentum pulled it back down and resulted in around 85% accuracy.
    * Decreasing learning rate lowered accuracy (not enough momentum)

The 2 simplest means to reduce overfitting are augmenting data and dropout. Which is what I did to solve the accruacy problem. I was able to increase the learning rate as well in this process to make the model more agressive, but not diverge.
    
My final model results were:
* Test Accuracy = 0.912
* Validation Accuracy = 0.934
* Training Accuracy = 0.953

If you graph the accuracy over iterations, the model converges amongst datasets and fits nicely:
![](markdown_data/valid_training.png?raw=true)

Architecture design decisions:

* The LeNet architecture was used as it suits small datasets and is fast in relative terms. It is designed for reading characters and contrasting patterns, such that signs are perfect candidates. It's gradient based so as classifications accruacy improves, robustness improves--as signs physically don't change much.

* What were some problems with the initial architecture?

The LeNet architecture didn't handle over and under fitting. Since the supplied dataset exhibited overfitting, I had to modify the architecture to support that state. That included adding dropouts, data augmentation and adjusting hyperparameters. Pooling and flattening were also added, but typically used in the LeNet architecture.

* Which parameters were tuned? How were they adjusted and why?

I wanted convergence as soon as possible since I did not have a GPU computer. So I was able to adjust learning rate to 0.002 and reduce batch size to 96 to get my result under 30 iterations/epochs.

Augmentation was the most helpful in keeping the accuracies from diverging as well as give me more "room for error" when adjusting the hyperparameters, for example increasing learning rate without extreme effects. With the UNaugmented dataset, dropout woud be too sensitive to adjust. More samples allows the architecture to be more robust. With enough data (over 100K samples) I was able to get rid of overfitting using the default LeNet parameters (but only achieved a 91% accuracy).
 

### 5. Test a Model on New Images

#### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![](markdown_data/test_data.jpg)

![](test_data/A.jpg)
![](test_data/B.jpg) 
![](test_data/C.jpg)
![](test_data/D.jpg) 
![](test_data/E.jpg)

#### Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

* For Image E.jpg, expected: 12     Got [12] (Priority vs. Priority)
* For Image D.jpg, expected: 25     Got [30] (Road work vs. Beware of ice/snow)
* For Image C.jpg, expected: 16     Got [20] (Vehicles over 3.5 metric tons prohibited vs Dangerous curve to the right)
* For Image B.jpg, expected: 1      Got [1]  (30km/h vs 30km/h)
* For Image A.jpg, expected: 27     Got [31] (Pedestrians vs Wild animals crossing)
 
The Test Accuracy = 0.600 (60%)

With an accuracy of 60% I think the next step is to add more augmented data and improve image quality (e.g. cropping, position, sharpening)

As for looking at the softmax probabilities of each image I found the following. In some ways the softmax results indicate we need more randomization on my dataset.

Priority sign:
  Priority         No Entry         No passing 3.5t  Double curve     Road narrow on right
  9.99999523e-01   4.24258218e-07   3.48734552e-08   5.21614363e-10   1.17968690e-10

Road work:
  Beware Ice       Road Work        Right of way     Turn left        Road narrow on right
  0.60561752       0.38658997       0.00324332       0.00209263       0.00092445

Vehicles over 3.5mt:
  Road Work         Dangerous curve   Roundabout       Yield           Stop      
  0.94134355        0.03380609        0.01610352       0.00469067      0.00296442
 

30km/hr:
  30km/h           100km/h          80km/h           50km/h           70km/h
  9.99997139e-01   1.43012130e-06   1.30631202e-06   1.04868008e-07   7.95507604e-10

Pedestrians:
Wild animals      General Caution     Double Curve      Slippery Road     Right of way
 0.93890363       0.03522851          0.01881449        0.00472944        0.00108749

![](markdown_data/softmax.jpg)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


