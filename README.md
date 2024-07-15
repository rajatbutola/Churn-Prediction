# Customer Churn Prediction
This project aims to predict customer churn using a dataset containing information about customers, their demographics, and their account information. The model is built using TensorFlow and Keras, and it employs a neural network to classify whether a customer will churn or not.


## Table of Contents
1. [Installation](#1-installation)
2. [Usage](#2-usage)
3. [Model Building](#3-model-building)
4. [Training Process](#4-training-process)
5. [Results](#5-results)
6. [Contribution](#6-contribution)

## 1. Installation
To run this project, you need to have the following libraries installed:

- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- keras
You can install these libraries using pip:

``` bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```
## 2. Usage
- ### 2.1. Upload the dataset

- The dataset used is `Churn_Modelling.csv`. Make sure to upload it in your Google Colab environment or local environment.
  
- ### 2.2. Run the code

- Use the code file `Churn_Prediction.ipynb` to load the data and run the program.

## 3. Model Building
- The neural network used in this project consists of:

- An input layer with optimum neurons
- Two hidden layers with optimum each, using ReLU activation
- An output layer with 1 neuron, using sigmoid activation
- The model is compiled with binary cross-entropy loss and the Adam optimizer.

## 4. Training Process
- In the training phase the model is trained with the appropriate number of epochs with a validation split of 20% from the training data to monitor overfitting.
- The hyperparameters are tuned such as learning rate, batch size, and number of epochs to further optimize model's performance.
- Training shows that both training and validation accuracies improved over epochs, indicating that the model is learning well from data.

## 5. Results
- The loss curves and accuracy curves are plotted to visually inspect model's performance over epochs and detect issues like overfitting.
  
## 6. Contribution

Contributions are welcome! Please submit a pull request or open an issue for any feature requests or bug reports. For major changes, please open an issue first to discuss what you would like to change.
