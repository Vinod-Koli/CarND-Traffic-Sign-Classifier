# **Traffic Sign Recognition** 
---

This project uses simple Convolutional Neural Network to classify German Traffic Signs.

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./DataSummary.png "Visualization"
[image2]: ./New_Images/3.jpg "Traffic Sign 1"
[image3]: ./New_Images/13.jpg "Traffic Sign 2"
[image4]: ./New_Images/23.jpg "Traffic Sign 3"
[image5]: ./New_Images/27.jpg "Traffic Sign 4"
[image6]: ./New_Images/31.jpg "Traffic Sign 5"

---
### Data Set Summary & Exploration

#### 1. Basic Summary of the Data Set

I used the `numpy` library to calculate summary statistics of the traffic signs data set:

* The size of training set: 34799
* The shape of a traffic sign image: 32x32x3
* The number of unique classes in the data set: 43

#### 2. Visualization of the dataset

Here is an exploratory visualization of the data set. It is a bar chart showing the number of training samples of each class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing the Data Set

The image data has been normalized so that the data has mean zero and equal variance. The simple and quick normalization technique used in this project is subsract each pixel value with 128 and divide by 128 as shown below,

`(pixel - 128)/ 128`


#### 2. Model Architecture
The architecture is based upon LeNet Model, with an additional layer of `dropout()`. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x3     	| 1x1 stride, VALID Padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x3	    | 1x1 stride, VALID Padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| outputs 400 				|
| Fully connected		| 400 to 120        									|
| Dropout		|         									|
| RELU					|												|
| Fully connected		| 120 to 84        									|
| RELU					|												|
| Fully connected		| 84 to 10        									|
 


#### 3. Training the Model

I have used Adam optimizer to train the model. The hyper parameters which best worked for me are as below
>Learing Rate = 0.00092,
>Epochs = 38,
>Batch Size = 128

#### 4. Approach towards acheiving 93% accuracy

From LeNet architecture I was able to get the accuracy of around 85%. Further I thought of avoiding overfitting by using `dropout` method. and also reduced `learning rate` and increased `epochs` to achieve even higher accuracy of 93%.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6] ![Lanes Image](./New_Images/38.jpg) 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model did a good job on new set of images, it was able to predict 5 images correctly out of 6.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield   									| 
| Slippery Raod     			| Slippery Raod 										|
| Pedestrians					| Pedestrians											|
| Speed Limit(60Km/h)	      		| Speed Limit(30Km/h)					 				|
| Wild Animal Crossing			| Wild Animal Crossing      							|
| Keep Right			| Keep Righ      							|

The model's accuracy was about 83%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


