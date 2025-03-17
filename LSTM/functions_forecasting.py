import numpy as np

# Function for recursive multi-step forecasting using a trained model
def recursive_multistep_forecasting(X_test, model, forecast_steps):
    """
    Perform recursive multi-step forecasting using a trained model.

    Parameters:
    - X_test: pandas DataFrame, testing features
    - model: trained machine learning model
    - forecast_steps: int, number of time steps to forecast into the future

    Returns:
    - preds: list, predicted values for each time step in the forecast_steps
    """
    # The example consists of the last observed values seen
    # In practice, it is the first example from the test set
    example = X_test.iloc[0].values.reshape(1, -1)

    preds = []
    for i in range(forecast_steps):
        pred = model.predict(example)[0]
        preds.append(pred)

        # Discard the value from the first position of the feature vector
        example = example[:, 1:]

        # Add the predicted value to the last position of the feature vector
        example = np.append(example, pred)
        example = example.reshape(1, -1)

    return preds