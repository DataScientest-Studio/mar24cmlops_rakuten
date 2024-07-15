def accuracy_from_df(df, prediction_col, true_col):
    """
    Calculate the accuracy score of the model's predictions.

    Args:
    - df (pd.DataFrame): The DataFrame containing the predictions and true categories.
    - prediction_col (str): The name of the column with the model's predictions.
    - true_col (str): The name of the column with the true categories.

    Returns:
    - float: The accuracy score.
    """
    # Compare the predictions with the true categories
    correct_predictions = df[prediction_col] == df[true_col]
    
    # Calculate the accuracy
    accuracy = correct_predictions.mean()
    
    return accuracy