##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/spatialcar.png
[image2]: ./examples/NotCarSpatial.png
[image3]: ./examples/colorhistcar.png
[image4]: ./examples/colorhistnotcar.png
[image5]: ./examples/hogy.png
[image6]: ./examples/hogcr.png
[image7]: ./examples/hogcb.png
[image8]: ./examples/original_test_image.png
[image9]: ./examples/original_test_with_grid.png
[image10]: ./examples/original_test_identified_grid.png
[image11]: ./examples/original_test_identified_heat.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In order to detect a vehicle in an image, a computer first must know what a vehicle is. This is accomplished by training a computer on features of a vehicle. We are able to extract various features from training images and feed them into a classifier that will be trained to make accurate prediction between vehicle and non-vehicle images. There are three features that I extracted from the training images that I will discuss below:

#####Color Histogram

One feature that we can extract from an image is the distribution colors within the image. Getting a histogram of the count of varying pixel colors can give us a feel for what is in the image. For instance, if there were a lot of sky blue pixels in an image, one could make an educated guess that the image will be a picture of the sky. The same can be applied to cars, where certain pixel color distributions can be used as a feature in training the computer on what a vehicle is.

Using the YCrCb color space I found the following histograms for a car and non-car image:

![alt text][image3]
![alt text][image4]

#####HOG

In detecting vehicles, obviously color can vary widely from vehicle to vehicle, but vehicles tend to be similar in their shape. Using edges or gradients is a great way to identify the shape of an object in an image. However, taking into account the different perspectives a vehicle could be viewed from, there is a wide variety of shapes that we can see. In order to decrease sensitivity in the variability of these shapes, we can divide the image into subsections and take a histogram of the the sum of the gradient directions and magnitude within the subsection. By doing this we can get a useful idea of the shape within the image, while maintaining the ability to allow noise and variation. Essential by doing this, our classifier will be able to get a much broader idea of the shape of a vehicle, given any perspective angle, and will accomadate for noise. I accomplish this in my code in lines 135 to 151 in train_classifier.py by using the hog function supplied by the skimage feature library, providing it parameters defined in vehicle_detector_variables.py. 

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image5]
![alt text][image6]
![alt text][image7]

#####Spatial Binning / Raw Pixels

Although in the real world, vehicles that we see won't match up exactly to the images in our training set, the raw pixels of a training image carries useful data to a classifier. However, a full resolution image in a 3 dimensional array will provide an overwhelming amound of features to process and bog our processing speeds down. Using spatial binning, we resize the image to a 16 X 16 pixel image and then unravel it from a 3 dimensional array to a vector that can be included into a feature list for classifier training. Below is a plot of the values in our car and non-car binned raw pixel features.

![alt text][image1]
![alt text][image2]


####2. Explain how you settled on your final choice of HOG parameters.

My final choice in HOG parameters, which can be seen in vehicle_detector_variables.py, were:

```
color_space = 'YCrCb'
orient = 9 
pix_per_cell = 8
cell_per_block = 2 
```

This was influenced strongly by the lesson provided by Udacity, and also by [section 6 of this paper](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf). I did empirically sample a few other parameter settings, but ultimately made this decision based on the prediction accuracy of the trained classifier that used these features.


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The following occurs in line 47 to 85 of train_classifier.py. All of the above described features are concatenated into a vectors for both cars and non cars, and then normalised by sklearn's StandardScaler function. We take the normalized features, and then shuffle and split the data into training and testing datasets. On line 76 we use a Sklearn's LinearSVC function as our support vector machine classifier and train this classifier on line 79. The classifier was trained and tested on 17,759 images with a feature vector length of 5,520 per image, achieving an accuracy of 0.9893018 in 15.1 seconds.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In order to find vehicles instead of stepping through the image and getting hog features for each sub-image one-by-one, we calculate hog features for the entire region-of-interest in the image and then subsample from the hog feature array. I do this in its entirety in the find_cars (lines 21-99) function in vehicle_detection.py. Using the same hog parameters that we used in our classifier training (stored in vehicle_detector_variables.py) I get hog features for each color channel within a specified region of interest of the y axis. We then step through the region of interest in both the X and Y axis, and for each step we take the hog features of the 3 different color channels, unravel them into a vector and stack them together to come up with a feature vector that is similar to what our classifier had been trained to make predictions with. In order to get our classifier to make accurate predictions we also need to include the spatial and color features of the current subsample of the image. With a feature vector including the hog, spatial, and color features of the current image subsample, we feed this feature vector to our SVC classifier's predict function, which will classify true if a vehicle is detected. In order to reduce false positives, I used the classifier's decision_function to get a confidence level of above .4 before decided. If the classifier detects the subsample of the image as a vehicle, with a confidence level of above .4, then I mark that subsample as a detected vehicle.

In order to maximize my confidence of detected vehicles, I did the above explained process multiple times per image on different scales, since cars will appear in different sizes in the video. toware the horizon, I used smaller scales with a smaller region of interest, and toward the bottom of the image I used larger scales over a larger region of interest. Each of these subsamples overlapped eachother by 75%. This took a lot of trial and error for me, trying to maximize vehicle detection and minimize false positives. I did this empirically, first with a handful of images, and then with sample videos. After a long and tedious process of trial and error, I found that the below scales and regions of interest worked best for my purposes.

Window Size | Region (Y Range)
------------ | -------------
(96, 96) | [390, 510]
(112, 112) | [400, 550]
(128, 128) | [460, 800]
(192, 192) | [460, 750]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Given the scales and regions mentioned above, the find_cars function in vehicle_detection.py detects vehicles as illustrated:

![alt text][image8]

![alt text][image9]

![alt text][image10]

In order to optimize the performance of vehicle detection, as mentioned above I performed HOG feature extraction on an entire region of interest and subsampled the extraction, rather than making hundreds of smaller hog extractions. Also I used carefully chosen regions of interest to only focus on parts of the image where the scaled sample would be about the size of a vehicle considering perception at a distance.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/qLcGN5ZHdVo)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In our vehicle detection on each image, we subsample repeatedly with multiple scale, and intentionally overlapping, we will end up with alot of overlapping detection areas within the image. By assigning a value of 1 to the area of each detection, we can add all of these detection together and create a heatmap of the entire image:

![alt text][image11]

We get the heatmap of each image by calling the add_heat function (located on line 148) in vehicle_detection.py, which iterates through our list of detects subsample boxes and adds a value for each box. Although we can do thresholding right away to remove noise from this image, I pass the heatmap to the average_heat_map function on line 100 of vehicle_detection.py, which combines the last 10 image heatmaps and thresholds them at a higher standard. This allows us to remove noise from object that span more than a few frames. 

From this heatmap, I use Scipy's Label function, which detects concentrated groupings of pixels, and seperately stores these blobs into a list. With these blobs, I could assume that these were confident detections of vehicles, and I pass this list to draw_labelled_bboxes, which iterates through the list and creates a box out of each blob's minimum and maximum x and y values. 

Once I have a list of boxes to draw, I use a global list of Vehicle objects. If the midpoint of the current box is within 100 pixels of a previously found Vehicle's box midpoint, then I add the current box to the vehicle's list detected boxes, which is averaged to get a final box. If none is found, I create a new vehicle, which won't appear until the vehicle is detected at least 30 times. This further removes the possibilty of false positives, but the averaging also causes a smooth tracking rather than jittering boxes in the video.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Since much Udacity provides resources to get me started with training the classifier and detecting vehicles in the images, I had little problem with detecting vehicles in a single image. However, with videos, I found it difficult to reduce all false positives while maintaining vehicle detections smoothly the entire time. There were certain spots in the project video where my video would detect a railgaurd as a car, and since the rail gaurd is in the video for 1-2 seconds, thats 30 to 60 frames where false positives are occuring. To reduce this I worked a lot on the scales I used in my vehicle detection function to minimize the false positives while optimizing vehicle detections. I also used heatmap averaging, and created a high standard of the number of vehicle detections before displaying the vehicle objects. All of the thresholding and averaging removes all false positive, but vehicles are slowly detected since there is such a high threshold. I had difficulty in balancing the tradeoff between responsiveness of the detection and false positive reduction.

Although I effectively detect vehicles in this video, I am afraid that I tuned everything for this video and that it is overfitted for this project. I am sure there is a better combination of parameters and scales that have been used. Especially in the case of performance. This code takes about 10-20x of the real-time of this video to process, which obviously will not work. Although there are likely enhancements I could have used to gain on performance, using a compiled language such as C++, we could achieve much improved performance.

