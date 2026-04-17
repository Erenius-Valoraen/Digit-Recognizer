# Digit-Recognizer

This project uses pytorch to make a simple Neural Network to classify hand drawn digits.

Model Architecture - 

```
Conv2D 1 (With Maxpool2D)
|
v
Conv2D 2 (With Maxpool2D)
|
v
Dropout2D
|
v
Fully Connected Layer 1
|
v
Dropout
|
v
Fully Connected Layer 2
```

All layers use ``` ReLU``` as their activation function and we use ```CrossEntropyLoss``` for the model. The final output is then passed through a `Softmax` operation to get probilities for all the digits.

Then, we use a streamlit canvas for getting input from the user and then process the user input to match the format expected by the model.
