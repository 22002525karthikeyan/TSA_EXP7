### Name: KARTHIKEYAN R
### Reg.No: 212222240046
### Date: 
# Ex.No: 07                                   AUTO REGRESSIVE MODEL
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model.
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

file_path = '/content/Gold Price.csv'
df = pd.read_csv(file_path)

df['date'] = pd.to_datetime(df['Date'])
df.set_index('date', inplace=True)


default_column_name = df.columns[1] 
series = df[default_column_name]
print(f"Using '{default_column_name}' for analysis.")


adf_result = adfuller(series)
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

# Train-test split (80% train, 20% test)
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(series, lags=30, ax=plt.gca())
plt.subplot(122)
plot_pacf(series, lags=30, ax=plt.gca())
plt.show()

# AutoRegressive model with specified lag
model = AutoReg(train, lags=13)
model_fitted = model.fit()

# Make predictions
predictions = model_fitted.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Plot Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.legend()
plt.title('AR Model - Actual vs Predicted')
plt.show()

# Calculate and display Mean Squared Error
mse = mean_squared_error(test, predictions)
print(f"Mean Squared Error: {mse}")

```
### OUTPUT:
#### GIVEN DATA
![image](https://github.com/user-attachments/assets/f7eb58d6-0d6b-440d-95e7-e1a4e7fdd3ea)

#### PACF - ACF
![Untitled](https://github.com/user-attachments/assets/5b977688-7b2d-42d3-929a-54b5fa9d2133)

#### PREDICTION
![Untitled](https://github.com/user-attachments/assets/cc1ec9e2-925d-4b2e-8cb9-e3c761e860bd)

#### FINIAL PREDICTION
![image](https://github.com/user-attachments/assets/086cc25d-361a-4268-8bf0-05c5ee54df19)

### RESULT:
Thus, the program for the implementation of the Autoregression function using Python was successfully completed.
