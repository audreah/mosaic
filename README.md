# mosaic

## Authors
Audrea Huang \<audreahuang@gmail.com> <br>
Péter Hámori \<hampet97@gmail.com>

## Description
Correctly identifying objects portrayed in images remains a significant challenge in computer vision, where existing models are trained on dominant figures within a contemporary context. Data bias of this form neglects underrepresented communities and variations of objects, thereby limiting extensions to more specific topics such as census reporting or photography. This project aims to expand upon existing models to recognize variations within common objects by providing a broader source of training images. The results of this work can be applied to aid in research methods for digital humanities and increase the efficiency of conducting inventory.

## Data Loading
Download 2014 Train/Val annotations from https://cocodataset.org/#download and name the folder "annotations_trainval2014"<br>
Run `coco_data_load.ipynb`

## Training and Evaluation
Store your path to `instances_train2014.json` in `annFileTrain`<br>
Run `coco_load_train_eval.ipynb`

## LeNet-5
We began this project by using the LeNet-5 model as a starting point. LeNet-5 was proposed by Yann LeCunn in 1989. It is a small and simple neural network, consisting of two Convolutional layers (each followed by an Average Pooling layer), a Flatten layer, and two or three Dense layers. 

Throughout the whole experiment with LeNet-5, we used images of the COCO dataset, which come with both labels and bounding boxes. The COCO dataset is thus suitable for both image classification and object detection tasks. Some of the most represented classes in the COCO dataset are person, car, chair, book, and bottle. For LeNet-5, we used the latter four. 

The size of the images of the COCO dataset vary between approximately 100x100 and 800x800. As neural networks take vectors of the same size as input, during preprocessing, we resized each image to 224x224. The reason we picked a relatively large picture size is to avoid losing too much information via compressing the pictures. However, this choice of size results in only 1200 images fitting the GPU provided by Google Colab. The 1200 images were randomly distributed into three disjunct sets for training, validation and test purposes. 

The images, along with their corresponding labels and masks, were downloaded via COCO’s API. As part of the preprocessing, images were normalized in the way we learned at class. 

Initially, we trained the original LeNet-5 for binary classification. Practically, this means asking the model if there is a car in the image. Here, the output is a scalar of either 0 or 1. 

Secondly, we performed multiclass image classification using the aforementioned four classes. We paid attention to the number of samples of each class. We did not feel the need to specify the numbers explicitly, as random sampling always resulted in a fairly balanced dataset containing 100 to 400 samples of each class. Additionally to the images (encoded as arrays), we also gave the model the masks as input as a four dimension after the red, green, and blue image channels. We figured specifying the place of objects within the image may help the prediction.

For this task, the target vectors are one-hot encoded, meaning they consist of four elements, which are exactly one 1, and three 0s. We used categorical cross entropy for the loss function, and Adam as optimizer. Each layer has a tangent hyperbolicus activation function, except for the Dense layer, which has Softmax, as we want the model to point out the most probable class. 

Throughout the semester we experimented with many different architectures. The final architecture of our model can be seen in the visualization below created in Netron. 

![LeNet-5 Network Visualization in Netron](https://github.com/audreah/mosaic/blob/main/LeNet-viz.png)
Figure 1. Network visualization of our LeNet-5 model.

We eliminated one Dense layer from the original LeNet-5 model, as well as decreased the number of parameters for the first Dense layer. As suggested by the Professor, we introduced regularization to the network (L2). We also applied dropout and early stopping. 
After performing manual hyperparameter optimization for the batch size, we opted for a relatively small number of 32. 
We did a detailed evaluation of the results using some of the metrics covered at class. The mean accuracy of the predicted target vectors over 5 runs is 0.38, while the precision is 0.31, the recall is 0.32, and the F1 score is 0.3. 

Taking a look at some of the predictions, we saw that most of the times the model seems to be indecisive between two classes, one of which is usually the correct class. 

Finally, we trained the modified version of LeNet-5 for multilabel image classification. This means that objects of each of the four classes may appear in the same image. For this task, we included images of multiple classes for the training as well. Here, the activation function of the last Dense layer is sigmoid, which produces probabilities for each class (each element of the target vector) individually. Afterwards, a threshold is defined. Elements of a predicted target vector with a value lower than the threshold are classified as 0, and higher values are considered 1. During our research, we read that different threshold values can be defined for the different classes, however, we did not do this because our dataset is not unbalanced. We used binary cross entropy as the loss function. 

The results of this task still have much room for improvement, as each element of the predicted target vectors are very close to 0.5, meaning that the model cannot decide if the given image contains some objects of that specific class. 

We think that the most reasonable improvement to this experiment would be to increase the training sample size. However, this is limited by the size of the GPU provided by Google Colab. 

## YOLOv4
Following our work with LeNet-5, we looked towards more advanced models such as YOLOv4 (You Only Look Once), a one-stage object detection model that incorporates convolutional layers and data augmentation to predict bounding boxes. The baseline model was trained on 80 classes within the COCO dataset but lacked accuracy in some areas due to the broad training data and categorizations. As such, this project utilizes transfer learning to focus the training on more specific labels. To achieve this, we included additional images from Google’s Open Image Dataset with objects from the classes “Book,” “Bottle,” “Car,” and “Chair” to match the objects learned with the LeNet-5 model. We chose this dataset because it presented a wide breadth of images with bounding boxes already drawn for training purposes. We also included the “Person” class in the YOLOv4 training data to widen the range of images to include another heavily represented label. The direct effects of doing so can be seen with the following image comparison, where the left image shows predictions from the baseline model and the right image displays bounding boxes from our specialized model.

![Predictions from base model](https://github.com/audreah/mosaic/blob/main/yolo-base-model.png)
![Predictions from base model](https://github.com/audreah/mosaic/blob/main/yolo_best_weights.png)
Figure 1. Left: predictions using baseline YOLOv4 model. Right: predictions from the retrained model, where the person on the left is correctly identified and the “chair” label for the stand in the middle is accurately removed.

The training set consisted of 2500 images per class (12,500 images total) whereas the validation and test sets were 20% of that size. Thus, we adjusted steps to 8000, 9000. Each input image was resized to 416 x 416 to accommodate for limitations of RAM and GPU usage on Google Colab. Since we focused on 5 classes, we set max_batches to 10000 and used 30 filters for each \[convolutional] layer preceding a \[yolo] layer. We adhered to recommendations from the authors of the baseline model to decide these hyperparameters. The model allows us to specify an ignore threshold, below which predictions would not be shown; we used a relatively low threshold of 0.3 to encompass most of the conclusions drawn in order to analyze how well the model was working. Over 10 hours of training produced a model with the following metrics for a confidence threshold of 0.25:

⋅⋅⋅precision = 0.38
⋅⋅⋅recall = 0.62
⋅⋅⋅F1-score = 0.47
⋅⋅⋅average IoU = 28.52 % 
⋅⋅⋅mean average precision (mAP@0.50) = 0.405164

These metrics were gathered from our YOLOv4 model pre-trained on the COCO dataset and re-trained on images from Google’s Open Image Dataset. Clearly, there is still room for improvement that could be achieved with more training, images of higher resolution, and a more diverse training dataset. 

