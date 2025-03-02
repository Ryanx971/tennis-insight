import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, precision_score, recall_score, f1_score
from scipy.stats import randint

logger = logging.getLogger(__name__)


def evaluate_additional_metrics(model, X_test, y_test):
    """
    Evaluate the model using Log Loss, Brier Score, Precision, Recall, and F1 Score.

    Returns:
        dict: Evaluation metrics.
    """
    probs = model.predict_proba(X_test)
    predictions = model.predict(X_test)
    ll = log_loss(y_test, probs)
    bs = brier_score_loss(y_test, probs[:, 1])
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    logger.info("Log Loss: %.4f", ll)
    logger.info("Brier Score: %.4f", bs)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall: %.4f", recall)
    logger.info("F1 Score: %.4f", f1)

    return {"log_loss": ll, "brier_score": bs, "precision": precision, "recall": recall, "f1": f1}


def tune_random_forest(X_train, y_train):
    """
    Perform randomized hyperparameter tuning for RandomForestClassifier using TimeSeriesSplit.

    Returns:
        best_model: The best RandomForestClassifier found.
    """
    param_dist = {
        "n_estimators": randint(50, 300),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
    }
    rf_base = RandomForestClassifier(random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)
    random_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=20,
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        scoring='accuracy'
    )
    random_search.fit(X_train, y_train)
    logger.info("Best hyperparameters: %s", random_search.best_params_)
    return random_search.best_estimator_


def train_match_model(df):
    """
    Train a RandomForest model using a time-based split (train on data before 2024, test on 2024),
    perform hyperparameter tuning, evaluate additional metrics, and save the optimized model.

    Returns:
        model: Trained RandomForestClassifier.
    """
    df = df.sort_values(
        by=["tourney_year", "tourney_month"]).reset_index(drop=True)
    train_df = df[df["tourney_year"] < 2024]
    test_df = df[df["tourney_year"] >= 2024]

    y_train = train_df["label"]
    X_train = train_df.drop(columns=["label"])
    y_test = test_df["label"]
    X_test = test_df.drop(columns=["label"])

    logger.info("Starting hyperparameter tuning for RandomForest...")
    best_rf = tune_random_forest(X_train, y_train)
    logger.info("Hyperparameter tuning completed.")

    predictions = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info("Time-based test set accuracy: %.4f", accuracy)

    additional_metrics = evaluate_additional_metrics(best_rf, X_test, y_test)
    logger.info("Additional evaluation metrics: %s", additional_metrics)

    # Save the trained model
    # model_file_name = 'random_forest_tennis_model.pkl'
    # directory = os.path.abspath('./models')
    # os.makedirs(directory, exist_ok=True)
    # model_path = os.path.join(directory, model_file_name)
    # joblib.dump(best_rf, model_path)
    # logger.info("Saved tuned RandomForest model to %s", model_path)

    return best_rf


def predict_match_outcome(classifier, features):
    """
    Predict the outcome of a match given match features.

    Args:
        classifier: Trained RandomForestClassifier.
        features (dict): Dictionary of match features.

    Returns:
        array: Prediction probabilities.
    """
    match_data = pd.DataFrame([features])
    return classifier.predict_proba(match_data)
