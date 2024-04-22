This Folder Containes Multilayered Perceptron trained with Backpropagation and tested on two sets from the UCI Machine Learning repository
IRIS and WINE datasets 

The difference is one includes normalization for feature scaling and other one does not include differences. 

User Input parameters from console: 
Input size (the number of features to be added as input to the model)
Number of hidden layers:
     Number of neurons at each layer
Number of output features (no of categories/classifications) expected 
Activation function (sigmoid or tanh)
Batch size
Number of epoches 
Learning Rate
Momentum 
Dataset choice (iris/wine)

For IRIS dataset ideal values are as follows :
Input size (the number of features to be added as input to the model) - 4
Number of hidden layers: (1/2) depending on choice 
     Number of neurons at each layer -  any between 1-9 for good results
Number of output features (no of categories/classifications) expected  - 3
Activation function (sigmoid or tanh) - depends on choice and requirement
Batch size = 32 ideal size but can vary for experimenting
Number of epoches  - 100 ideal but consider more than that  i have tested till 1000 epoches 
Learning Rate - 0.01 ideal case
Momentum - 0.99 ideal case
Dataset choice (iris/wine) - iris 

For WINE dataset ideal values are as follows :
Input size (the number of features to be added as input to the model) - 13
Number of hidden layers: (1/2) depending on choice 
     Number of neurons at each layer -  any between 1-9 for good results
Number of output features (no of categories/classifications) expected  - 3
Activation function (sigmoid or tanh) - depends on choice and requirement
Batch size = 32 ideal size but can vary for experimenting
Number of epoches  - 100 ideal but consider more than that  i have tested till 1000 epoches 
Learning Rate - 0.01 ideal case
Momentum - 0.99 ideal case
Dataset choice (iris/wine) -wine
