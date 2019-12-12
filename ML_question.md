# Part one:

### Data Engineering:
=====================

+ How to deal with missing data
+ How to detect & deal with outliers
+ How to deal with imbalanced data
  - Data augmentation for image data
+ Resampling method


### Feature Engineering:
______________
- How/why to deal with thousands of features
  + Ways to perform dimension reduction
   - Explain PCA
- How to deal with categorical variables.
- How/why we scale/normalize features

### Modeling
______________
```
class ML_Model:
    def __init__(self):
        self.params = ? # what params is included, how to initialize it
        self.learning_rate # where this params is used?
        
    def predict(self, X):
        Y_pred = func(X, self.params)
        return Y_pred
  
    def loss(self, Y_pred, Y_true):
        loss = func(Y_pred, Y_true) # what is func
        return loss

    def update_params(self, loss, learning_rate):
        params_new = func(loss, self.params) # what is func
        return params_new
        
    def train(self, X, Y_true, epochs):
        for i in range(epochs):
            Y_pred = self.predict(X)
            loss = self.loss(Y_pred, Y_true)
            self.params = self.update_params(loss, self.params)
        return self.params
            
```
#### General:
- Theory
  - Explain term: parametric model & non-parametric model, what is the difference
  - Explain model(any) to a kid

- Optimization & evaluation:
  - Explain variance and bias
  - How to control fitting for all models    

#### Linear regression:
- Assumption:
  - What is the assumption behind linear regression.
  - How to detect/process multi-colinearity
  - What is / how to use VIF

- Formula:
  - what is the formula to calculate Y
  - Explain activation function, why it works

+ Evaluation & optimization:
  - Loss function: What is the loss function
  - Over fitting: how to detect, control

+ Feature selection:
  + Methods to perform feature selection and theory behind it.
    
  
