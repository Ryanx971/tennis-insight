import os
import joblib
import logging

from sklearn.preprocessing import LabelBinarizer
from . import features_service
from . import model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE_NAME = 'random_forest_tennis_model.pkl'
MODEL_DIRECTORY = os.path.abspath('./models')
MODEL_PATH = os.path.join(MODEL_DIRECTORY, MODEL_FILE_NAME)


def model_exists():
    """
    This function checks if a model file exists at the specified path. 
    It returns True if a model file is found, indicating that the model has been previously saved.
    Otherwise, it returns False.

    Returns:
    - bool: True if a model is saved at the specified path, False otherwise.
    """
    return os.path.exists(MODEL_PATH)


def load_or_train_model():
    """
    Load the model if it exists, otherwise train a new one.

    Returns:
    - RandomForestClassifier: A trained random forest classifier.
    """
    if model_exists():
        logger.info("Model exists, loading existing model.")
        classifier = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.info("Model does not exist, training a new model.")
        data = model_service.load_data()
        df = model_service.preprocess_data(data)
        df = model_service.create_label(df)
        df = model_service.encode_categorical_features(df)
        df = model_service.handle_missing_values(df)
        classifier = model_service.train_model(df)
        joblib.dump(classifier, MODEL_PATH)
        logger.info("Model trained and saved successfully.")
    return classifier


def get_match_features():
    """
    Extract match features from environment variables.

    Returns:
    - dict: A dictionary containing match features.
    """
    encoder = LabelBinarizer()
    match_features = {
        'surface': encoder.fit_transform([os.getenv('SURFACE')])[0],
        'tourney_level': encoder.fit_transform([os.getenv('TOURNEY_LEVEL')])[0],
        'best_of': int(os.getenv('BEST_OF')),
        'round': encoder.fit_transform([os.getenv('ROUND')])[0],
        'tourney_year': int(os.getenv('TOURNEY_YEAR')),
        'tourney_month': int(os.getenv('TOURNEY_MONTH')),
        'first_player_seed': int(os.getenv('FIRST_PLAYER_SEED')),
        'second_player_seed': int(os.getenv('SECOND_PLAYER_SEED')),
    }
    return match_features


def predict(player1, player2):
    """
    Predict a tennis match between two players.

    This function loads ATP tennis match data, preprocesses it, trains a random forest classifier
    to predict match outcomes, extracts relevant features for both players, merges them with match
    features, and predicts the outcome of the match. It then prints the probability of each player
    winning the match.

    Args:
    - player1 (str): The name of the first player.
    - player2 (str): The name of the second player.

    Returns:
    - dict: A dictionary containing the predicted win probabilities for each player.
    """
    classifier = load_or_train_model()

    player1_features = features_service.extract_features(player1)
    player2_features = features_service.extract_features(player2)
    match_features = get_match_features()

    features = features_service.merge_match_and_player_features(
        match_features, player1_features, player2_features
    )

    logger.info("Features for this match: %s", features)

    prediction = model_service.predict_match(classifier, features)
    player1_win_prob = round(prediction[0][1] * 100, 2)
    player2_win_prob = round(prediction[0][0] * 100, 2)

    logger.info("Prediction for the match:")
    logger.info("%s: %.2f%%", player1, player1_win_prob)
    logger.info("%s: %.2f%%", player2, player2_win_prob)

    return {
        player1: player1_win_prob,
        player2: player2_win_prob
    }
