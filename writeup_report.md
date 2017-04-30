#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

For my model, I am using the Network Architecture published by the Nvidia team. Which consists of normalization layer followed by five convolution layer and finally three fully connected layer.  

####2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the center image but found out that the vehicle was leaning towards one side of the road. Therefore I also flipped the image and multplied the steering angle by -1. Also I drove around the track couple of times to collect the data and turned the car around to gather the data driving the car around the track backwards.  

###Model Architecture and Training Strategy

####1. Solution Design Approach

After following the tutorial and implementing the Nvidia Archtecture, the car was predicting the angles pretty well. To improve and have the car be able to drive around the track without leaving the road, I had to crop the images, to get rid of parts of images that were irrelevant. On the top portion of the images, it capture big chunks of the sky, which I came up with the conclusion that it can be discarded and also the bottom portion of the image was made up with the hood of the car, which we discarded as well. 

We also normalized the images, and this significantly improved the results. Finally, we improved the overall data. The way we improved the data, was by driving around the track couple times, than driving around the track backwards couple of times, and finally recording data of turning sharp turns couple times as well. With these implementations, my car was able to drive around the track without leaving the road or crashing.  

####2. Final Model Architecture

The Model Architecture has five convolution layer followed by three fully connected layer that outputs the control value. The first three convolution layer is a convolution layers with a 2x2 strides and a 5x5 kernel. And finally the last two convolutional layer is non strided layer with 3x3 kernel. All these convolutional layers use relu activation.

After the convolutional layer it's followed by three connected layer that outputs the control value.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. After second lap, I stopped the car and drove around the track in other direction for two more laps.

After collecting the data, I flipped the image array and mutlplied the angle of the steering wheel by -1. This helped my car from leaning more towards one direction and also doubled the data I collected. 

After collecting the data, I shuffled the data and also split the data for training and testing. Where 20% of the data was set aside for the test.
