from sklearn.metrics import classification_report,confusion_matrix

def evaluate_model(model, x_test, y_test):
    """
    Evaluates an XGBoost model on test data
    

    Args: 
        model: Trained model.
        x_test: Test features.
        y_test: Test labels.
    """

    preds = model.predict(x_test)
    print("Classification Report:\n - evaluate.py:15", classification_report(y_test, preds))
    print("Confusion Matrix:\n - evaluate.py:16", confusion_matrix(y_test, preds))