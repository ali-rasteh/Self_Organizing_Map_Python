# Self_Organizing_Map_Python

Self Organizing Map Implementation in Python

## Description

This project implements a Self-Organizing Map (SOM) in Python. A SOM is a type of artificial neural network that is trained using unsupervised learning to produce a low-dimensional representation of the input space.

## Datasets

Several datasets are used to train and evaluate the SOM:

1. **Hayes-Roth**: An artificial dataset created to test prototype-based classifiers.
2. **Haberman's Survival**: Contains cases from a study on the survival of patients who had undergone surgery for breast cancer.
3. **Banana**: An artificial dataset with instances belonging to several clusters with a banana shape.
4. **Balance Scale**: Generated to model psychological experimental results, where each example is classified as having the balance scale tip to the right, left, or be balanced.

## Code Structure

- **SOM Codes/som.py**: Contains the implementation of the SOM and Node classes. The SOM class includes methods for training and predicting using the SOM.
- **SOM Codes/main.py**: Script to load datasets, train the SOM, and evaluate its performance on different datasets.

## Usage

1. **Training**: The SOM can be trained on various datasets by adjusting the parameters such as iterations, learning rate, and dataset-specific configurations.
2. **Prediction**: After training, the SOM can be used to predict the class of new input feature vectors.

## Requirements

- Python
- NumPy
- scikit-learn

## How to Run

1. Initialize the SOM with the desired parameters.
2. Train the SOM using the provided datasets.
3. Evaluate the SOM's performance on test data.

## Example

```python
from som import SOM

# Initialize the SOM
som = SOM(height=5, width=5, FV_size=2, PV_size=1, learning_rate=0.05)

# Train the SOM with a sample dataset (e.g., XOR function)
som.train(iterations=100, train_vector=[[[1, 0], [1]], [[1, 1], [0]], [[0, 1], [1]], [[0, 0], [0]]])

# Make predictions
print("Prediction for [0, 0]:", round(som.predict([0, 0])[0]))
print("Prediction for [1, 0]:", round(som.predict([1, 0])[0]))
```

For more details, refer to the code files in the `SOM Codes` directory.
