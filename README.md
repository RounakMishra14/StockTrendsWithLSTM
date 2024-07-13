# Stock Trends Prediction with LSTM

This project implements a Long Short-Term Memory (LSTM) model to predict stock prices using historical data. The model is built with TensorFlow and Keras, leveraging advanced data preprocessing, model tuning, and evaluation techniques to forecast stock trends with high accuracy.

## Project Overview

The primary objective of this project is to develop a robust predictive model for stock price movements based on historical data. Using Amazon (AMZN) stock data, the LSTM model predicts future stock prices by learning from past price trends and volume metrics.

## Technical Details

### Technologies Used

- **Python**: Utilized for scripting and data manipulation.
- **Pandas & NumPy**: For data handling and numerical computations.
- **Matplotlib & Seaborn**: Used for visualizations to understand data patterns and model performance.
- **TensorFlow & Keras**: Frameworks employed for building and training the LSTM model.
- **Scikit-learn**: For evaluation metrics such as RMSE, MSE, and MAE.

### Data Preprocessing

The dataset (`amzn_stock_data.csv`) undergoes extensive preprocessing steps:

- **Normalization**: Scaling numerical features to a standard range.
- **Sequence Generation**: Creating input sequences for LSTM training.

### Model Architecture

The LSTM model architecture includes:

- **Sequential Layers**: Stacked LSTM layers for capturing temporal dependencies.
- **Dropout Layers**: To prevent overfitting and enhance generalization.
- **Dense Layers**: For final output prediction.

### Evaluation Metrics

The model is evaluated using the following metrics:

- **Root Mean Squared Error (RMSE)**: Measures deviation between predicted and actual prices.
- **Mean Squared Error (MSE)**: Quantifies average squared differences.
- **Mean Absolute Error (MAE)**: Measures average magnitude of errors.

### Results and Analysis

After training over 50 epochs, the model achieves the following performance metrics:

- **RMSE**: 77.01
- **MSE**: 5930.79
- **MAE**: 25.41

### Visualizations and Insights

The project includes insightful visualizations such as:

![Pairplot of Stock Attributes](https://github.com/rounakmishra/Stock-Trends-LSTM/blob/main/images/pairplot.png)  
*Pairplot of stock attributes showing relationships between metrics.*

![Correlation Matrix](https://github.com/rounakmishra/Stock-Trends-LSTM/blob/main/images/correlation_matrix.png)  
*Correlation matrix analyzing correlations between prices and volumes.*

![Actual vs. Predicted Prices](https://github.com/rounakmishra/Stock-Trends-LSTM/blob/main/images/actual_vs_predicted.png)  
*Visual representation of model predictions versus actual stock prices.*

### Future Enhancements

To further enhance the project’s analytical capabilities and model performance:

- **Hyperparameter Tuning**: Optimize LSTM parameters for better accuracy.
- **Feature Engineering**: Incorporate sentiment analysis or macroeconomic factors.
- **Alternative Models**: Compare LSTM with other deep learning or statistical models.
