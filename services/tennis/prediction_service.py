import os
import logging
import joblib
from sklearn.preprocessing import LabelBinarizer
from services.tennis import features_service
from services.tennis import data_preprocessing_service
from services.tennis import model_training_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_FILE_NAME = 'random_forest_tennis_model.pkl'
MODEL_DIRECTORY = os.path.abspath('./models')
MODEL_PATH = os.path.join(MODEL_DIRECTORY, MODEL_FILE_NAME)


def is_model_available():
    """
    Check if a trained model file exists at the specified path.

    Returns:
        bool: True if a model is saved at MODEL_PATH, False otherwise.
    """
    return os.path.exists(MODEL_PATH)


def get_or_train_model():
    """
    Loads the model if available; otherwise, trains a new one.

    Returns:
        RandomForestClassifier: A trained RandomForest model.
    """
    if is_model_available():
        logger.info("Model exists, loading the existing model.")
        classifier = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.info("Model not found, training a new model.")
        data = data_preprocessing_service.load_match_data()
        df = data_preprocessing_service.preprocess_match_data(data)
        df = data_preprocessing_service.create_match_labels(df)
        df = data_preprocessing_service.encode_categorical_features(df)
        df = data_preprocessing_service.impute_missing_values(df)
        classifier = model_training_service.train_match_model(df)
    return classifier


def extract_match_features():
    """
    Extract match features from environment variables.

    Returns:
        dict: A dictionary containing match features.
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


def predict_match_outcome(player1, player2):
    """
    Predict the outcome of a tennis match between two players.

    This function loads (or trains) the model, extracts features for the match and both players,
    then predicts and logs the win probabilities.

    Args:
        player1 (str): Name of the first player.
        player2 (str): Name of the second player.

    Returns:
        dict: A dictionary with predicted win probabilities for each player.
              For example: {player1: 75.25, player2: 24.75}
    """
    classifier = get_or_train_model()

    player1_features = features_service.get_player_features(player1)
    player2_features = features_service.get_player_features(player2)
    match_features = extract_match_features()

    combined_features = features_service.merge_match_player_features(
        match_features, player1_features, player2_features
    )

    # logger.info("Combined features for the match: %s", combined_features)

    prediction = model_training_service.predict_match_outcome(
        classifier, combined_features)
    player1_win_prob = round(prediction[0][1] * 100, 2)
    player2_win_prob = round(prediction[0][0] * 100, 2)

    logger.info("Prediction for the match:")
    logger.info("%s: %.2f%%", player1, player1_win_prob)
    logger.info("%s: %.2f%%", player2, player2_win_prob)

    return {player1: player1_win_prob, player2: player2_win_prob}
