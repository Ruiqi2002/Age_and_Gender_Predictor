# Predicting the apparent Age and Gender using face images

# This project aims to implement:-


1) Preprocessing image

2) Perform Eigenface Analysis for Eigendecomposition and Singular Value Decomposition (SVD). Discuss their similarity and differences

3) Develop model to predict apparent age and gender using face images (SVD Processed feature) 

4) implement an Artificial Neural Network (ANN) 

5) discuss our experimental design

<a id = "Table_of_content"></a>
# Table of content
### 1. [Import Library](#import_library)
### 2. [Data Proprocessing](#Data_Proprocessing)
### 3. [EigenFace Analyse](#Eigenface_Analyse)
### 4. [Predict apparent age and gender using face images](#Predict_apparent_age_and_gender_using_face_images)
### 5. [ANN Model](#ANN_Model)
### 6. [Comparison (Linear Regression, Linear Regression using SVD, ANN with MSE , ANN with ADAM)](#Comparison)
### 7. [Conclusion](#Conclusion)
### 8. [Our Model Prediction](#Model_Prediction)

<a id = "import_Library"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Import Library
For this project, we have implemented these libraries:

|library      |Functions/Usage    |
|:--|:--|
|numpy          |- uses in creating arrays |
|               |- performs operations in array.etc|
|sys            |- manipulate different parts of Python runtime environment|
|skimage        |- use in image processing (greyscale, rescaling) |
|matplotlib     |- library use for ploting |
|glob           |- an Unix style pathname expansion for python |
|os             |- library to call upon os system|
|time           |- access time in system | 
|pandas         |- open source data analysis and manipulation tool |
|sklearn        |- library for model selection |

<a id = "Data_Proprocessing"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Data Preprocessing
this section preprocess data from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

Note: 
After checking , we found out that "61_3_20170109150557335.jpg.chip.jpg" contains an error thats its second value contains a "3"
The following image can lead to an missing data in the dataset.
Thus, we decided to remove this data from the dataset

Another solution is to add it to 62_1_.......jpg 


# In the code below, we will set:

D as our dataset <br >

y as the Age of Dataset image (transform into float value between 0 and 1)<br >
To transform the data, the formula:
$\displaystyle \frac{Age}{Max Age}$<br >
will be used

Example: <br >
$\displaystyle \frac{72.0}{100.0} = 0.655$ <br >
<br >
The answer will be in float type (decimal
<br >

z will be our Gender in Dataset, which will only contains 0 and 1 <br >
For the later computation, these data needs to be in the forms of -1s and 1s. <br >
# Note that, in “Gender”，
0 is Male<br >
1 is Female<br >

# To transform the data, we will be using the formula,
$2(Gender)-1 $, <br >

From this equation, we can make the values in Gender into -1 and 1 only. <br >
After that, we will print the shape of the list for debugging/checking purpose.

# Splitting the dataset into training and testing
The dataset will be split into two dataset
* training dataset
* testing dataset

# Why do we split the dataset into 90% train and 10% test? 
There is mainly due to the size of our dataset. Based on our description above, we can obtain informations of our dataset. <br>
Our dataset only have  __total of 9779 instances__. To obtain a better result, most data instances should be use to train the model. 

# To obtain fixed result, we have fix the random state as 205

<a id = "Eigenface_Analyse"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# EigenFace Analyse 
    
Eigenface is a name given to a set of eigenvectors when it is used in computer vision problems of human face recognition. <br>

This approach of using eigenfaces for face recognition were developed by Sirovich and Kirby. <br>

The Eigenface technique is using the space of images (face images) to project them as low demensional representation of face images. <br >
 
Using Principal Component Analysis (PCA), we can use a cpllection of face images to form a set of basis features. The determined principal components can be used to recosntruct an input image and classify a face as an element.

<a id = "Eigendecomposition_Vs_Singular_Decomposition(SVD)"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Eigendecomposition Vs Singular Decomposition（SVD）

Consider the formula of Eigendecompositon $A = PDP^{-1}$ and the formula of Singular Value Decomposition(SVD) SVD $A=UΣV^{T}$

| Description | Eigendecompostion | SVD |
| :--- | :--- | :--- |
|Properties| The Vector Matrix, $P$ are not necessarily orthogonal, thus, the change of basis can be rotation |The vectors $U$ and $V$ are orthonormal, so they can perform rotations |
|-|Matrix $P$ and $P^{-1}$ are inverse of each other|$U$ and $V$ are not necessarily inverse of each other |
|-|D can be any complex number|The entries in the diagonal matrix Σ arre real and positive value |
|-|Can only exist in square matrix due to its formula |Can be both rectangular or square matrix |
|Theory |SVD says for any linear map, there is an orthonormal frame in the domain such that it is first mapped to a different orthonormal frame in the image space, and then the values are scaled.|Eigendecomposition says that there is a basis, it doesn't have to be orthonormal, such that when the matrix is applied, this basis is simply scaled.<br><br> Assuming we have  𝑛 linearly independent eigenvectors of course. In some cases your eigenspaces may have the linear map behave more like upper triangular matrices. (not sure)|
Eigenface|The column of P is the Eigenface |The row of Vt is the Eigenface |

<a id = "Eigendecomposition"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Eigendecomposition

Eigendecomposition, also known as matrix diagonalization aims to  perform a matrix decomposition of a square matrix into eigenvalues and eigenvectors.

The eigenvectors are derived from the covariance matrix over high-dimensional vectie space of face images. Yhe eigenvectors will form a basis set of images that can use to constrcut covariance matrix which can define a new coordinate system:
* Eigenvectors with the largest eigenvalue has the most variation among the training vectors $x$<br>
* Eigenvectors with the smallest eigenvalue has the least variation <br>
* Thus, the eigenvector formed can be derived as (principle component/eigenface)

<a id = "Singular_Value_Decomposition(SVD)"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Singular Value Decomposition (SVD)
SVD can refactors the images into three different matrices, U S and VT. U and VT are known as singular vector and entry in Σ are singular values 

<a id = "Discussion_on_similarity_and_difference"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Dicsussion on similarity and difference
|Similarity|Difference|
|:--|:--|
|For both Eigen decomposition and SVD have produce similar faces |For Eigen decomposition, we are required to obtain the mean centered data of the image using $D – mean$.  <br>|
|Example, Eigenface 4,5,6,9 |After that, we calculate the covariance matrix and obtain the eigenvalue and eigenvectors.|
|This is because both equations are methods to reduce high dimensional images in dataset while retaining as many features in the data.|For SVD, we use $D^{T}D$ to obtain the eigenvalue and eigenvectors.|
| Thus, it is possible for them to obtain similar eigenfaces.||

<a id = "Preprocessing_for_SVD"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Preprocessing for SVD processed features
In this section, we will preprocess the dataset for step 5 - develop model using SVD process feature <br>
* Step 1: calculate the mean of the dataset 
* Step 2: Demean the dataset (x_train - x_train_mean)
* Step 3: Whittening the dataset 
> Note: Whitening is a data pre-processing step. It can be used to remove correlation or dependencies between features in a dataset. <br>

# Linear Regression
<a id = "Predict_apparent_age_and_gender_using_face_images"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
## Predict apparent age and gender using face images
For this question, we have choose to apply Closed Form Solution with/without regularization parameter(λ) from our lecture

The formula for closed form solution is: <br> 
$W = (D^TD+λI)^{-1}D^Ty $ <br> 

For better visualization, we will design an experiment to find out the most suitable model to run the dataset. <br>

The aim to create an experiment here to figure out the purpose and usage of SVD in Closed Form Solution and how does it affect our results. <br>

Before we continued in this section, we need to know that:
- Predicting Age is a regression problem. Thus, we should focus on Mean Squared Error (MSE)
- Predicting Gender is a classfication problem (only two output will be given). Thus, we should focus on accuracy score

# Simple conclusion on the linear regression

> Based on observation, we can conclude that the bigger the lambda, the longer the time taken to compute.

> The lower lambda value,lower mean sqaured error and the higher accuracy of Gender Prediction. 

# Among the 5 values we tested ,we can found out that 50 is the best hyperparameter for linear regression.

# Linear Regression using SVD
For this SVD, we will be using the whitening reduced dataset to perform the SVD Processed feature

# Simple Conclusion on the Linear Regression using SVD processed feature
When we compare this model to our linear regression model, we can spot some difference.<br >

One of the most significant differences will be: <br>
* the time taken to compute the model. Linear Regression using SVD processed feature quicker. 

In our features normalization part,we calculate the mean of the features first.This step is to center the data around the zero and remove the bias.Then we compute demean by substarct mean of features.Demean can ensure the data are around the zero and also handle bias removing.After that, we want to increase more performance of model then we let all the features in a similar scale by whitenning.
Target of those steps is to normalize the features by center data,remove bias, scaling features and ensure the models are not overfitting.This is the reason why time taken of linear regression using SVD decrease from 2 seconds to 0.1 seconds.

<a id = "ANN_Model"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# ANN(SGD and ADAM)

A prediction function will be written and used in both SGD and ADAM <br>
In the prediction function,
sigmoid function,

$\displaystyle \frac{1}{1 + \exp(-y)}$<br >

Both model will be using the whitten_reduced dataset for better efficiency

<a id = "Comparison"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Comparison between Linear Regression, Linear Regression using SVD, ANN with MSE and ANN with ADAM

<style type="text/css">
#T_a2631_row0_col0, #T_a2631_row0_col1, #T_a2631_row0_col2, #T_a2631_row0_col3, #T_a2631_row1_col0, #T_a2631_row1_col1, #T_a2631_row1_col2, #T_a2631_row1_col3, #T_a2631_row2_col0, #T_a2631_row2_col1, #T_a2631_row2_col2, #T_a2631_row2_col3, #T_a2631_row3_col0, #T_a2631_row3_col1, #T_a2631_row3_col2, #T_a2631_row3_col3 {
  text-align: center;
}
</style>
<table id="T_a2631">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_a2631_level0_col0" class="col_heading level0 col0" >Model</th>
      <th id="T_a2631_level0_col1" class="col_heading level0 col1" ></th>
      <th id="T_a2631_level0_col2" class="col_heading level0 col2" >MSE</th>
      <th id="T_a2631_level0_col3" class="col_heading level0 col3" >ACC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_a2631_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_a2631_row0_col0" class="data row0 col0" >Linear Regression</td>
      <td id="T_a2631_row0_col1" class="data row0 col1" >-></td>
      <td id="T_a2631_row0_col2" class="data row0 col2" >0.066193</td>
      <td id="T_a2631_row0_col3" class="data row0 col3" >0.668712</td>
    </tr>
    <tr>
      <th id="T_a2631_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_a2631_row1_col0" class="data row1 col0" >Linear using SVD</td>
      <td id="T_a2631_row1_col1" class="data row1 col1" >-></td>
      <td id="T_a2631_row1_col2" class="data row1 col2" >0.075199</td>
      <td id="T_a2631_row1_col3" class="data row1 col3" >0.643149</td>
    </tr>
    <tr>
      <th id="T_a2631_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_a2631_row2_col0" class="data row2 col0" >SGD</td>
      <td id="T_a2631_row2_col1" class="data row2 col1" >-></td>
      <td id="T_a2631_row2_col2" class="data row2 col2" >0.203906</td>
      <td id="T_a2631_row2_col3" class="data row2 col3" >0.537832</td>
    </tr>
    <tr>
      <th id="T_a2631_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_a2631_row3_col0" class="data row3 col0" >ADAM</td>
      <td id="T_a2631_row3_col1" class="data row3 col1" >-></td>
      <td id="T_a2631_row3_col2" class="data row3 col2" >0.078690</td>
      <td id="T_a2631_row3_col3" class="data row3 col3" >0.627812</td>
    </tr>
  </tbody>
</table>


# Discussion
* Throught observing the table,we can know that linear regression without SVD had the lowest MSE and highest ACC.This model use the orignal dataset and another models used the SVD processed features.Then there has a probabiity of SVD maybe remove some important features and quantity of train set decrease.Therefore another model maybe underfitting.
* Time Taken of Linear Regression with using SVD is less than model without SVD because of quantity of features linear model using svd is less than normal linear model.
* we can come to conclusion that SVD Process Feature plays a huge role in reducing the time taken to compute a model

### From the result of both ANN with SGD and ADAM

* For the SGD model and ADAM model,we can improve the efficiency of both model through modify the number of learning rate,number of nueron in hidden layer and epochs.
* We find the MSE gradually decrease and ACC gradually increase when the number of epochs increase.Therefore we set all number of epochs of the predictive model to be 30.Through our testing (learning rate,number of nueron in hidden layer),we use (0.01,3) for SGD model to predict age.(0.001,3) for SGD model to predict gender and ADAM model to predict age .(0.005,50) for ADAM model to predict gender.
* For the result,ADAM has a higher performance than SGD model.The trainset for predict gender of ADAM had a result which near to 0.9.This is a quite high result of predictive model.However SGD only had 0.5 acc of gender prediction.Moreover,SGD model had nearly 0.3 score of mse and result o ADAM almost less than 0.1.
* Althought ADAM is better than SGD but time process is moere than SGD model.
* Therefore we can know ADAM model is better than SGD model in age and gender prediction but maybe SGD model also can be improved by find the best hyperparameter for it.

# Conclusion
As a conclusion, we can know that linear regression without using svd is the best model topredict age and gender 

As the results, the 4 models have almost same result of trainset that mean the 4 models are such a good implementation.

<a id = "Model_Prediction"></a></a><div style="text-align: right"> <a href=#Table_of_content>Back?</a> </div>
# Our Model Prediction
In this section, we have implement the model on top to predict our own images

# Conclusion
Because of linear regression without using SVD processed features had the lowest MSE and highest ACC.Therefore we decided to use it for progress our selfie to predict age and gender.

After fitting data into the model,we get a good results for our picture because there is no exaggerate results to us.Among the result there is only the picture 3 has special result maybe that mean picture 3 had some problem.As the result can show that,linear regression without using SVD is a good implementation

<a align= "center">~ THANK YOU ~</a>


