# Implementation of ANN on Yale Dataset

Implemented a MultiLayer Perceptron (MLP) for facial recognition using the Yale dataset available at https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database

## Data Preprocessing
I loaded and preprocessed the Yale dataset, which comprises 165 GIF images of 15 subjects. Data preprocessing steps included resizing images, flattening them into 1D arrays, and extracting labels from file names.

## Data Splitting
I split the dataset into training and testing sets, allocating 80% of the data for training and 20% for testing. Stratified splitting was performed to maintain the class distribution in both sets.

## MLP Architecture
My MLP architecture consists of an input layer, one hidden layer with 128 neurons, and another hidden layer with 64 neurons, followed by an output layer with
15 neurons (corresponding to the 15 subjects). We used the ReLU activation function in hidden layers and softmax activation in the output layer.

## Model Training
I trained the MLP model for 60 epochs with a batch size of 64, using the Adam optimizer with a learning rate of 0.001. A 10% validation split was employed for monitoring model performance during training.

## Evaluation Metrics
To evaluate the model’s performance, I calculated accuracy, precision, recall, and F1 score on the testing dataset. These metrics provide insights into the model’s ability to recognize faces accurately.

## Visualization
I visualized the training and validation loss over epochs to assess the model’s training progress. Additionally, we generated a confusion matrix heatmap to visualize the model’s performance in classifying subjects. The methodology outlined above served as the foundation for my face recognition implementation using an MLP architecture.

## Hyperparameter Tuning
To enhance the performance of our MLP-based face recognition model, I conducted hyperparameter tuning using the GridSearchCV approach. GridSearchCV systematically tested combinations of these hyperparameters
using a 3-fold cross-validation approach. Our custom scoring metrics for accuracy, precision, recall, and F1 score were employed to identify the best configuration. After extensive hyperparameter tuning, the best configuration was
identified, yielding the following hyperparameters:
### Activation function: ’relu’
### Batch size: 64
### Number of epochs: 70
### Hidden layers: [128, 64]
### Optimizer: ’adam’

## Experimental Results
The model was evaluated on the testing dataset, and the following performance metrics were calculated:
### Accuracy: 0.8182 (81.82%)
### Precision: 0.8929 (89.29%)
### Recall: 0.8667 (86.67%)
### F1 Score: 0.8441 (84.41%)

