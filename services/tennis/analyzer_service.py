import os
import joblib

from sklearn.preprocessing import LabelBinarizer
from . import features_service
from . import model_service


def model_exists():
    """
    This function checks if a model file exists at the specified path. 
    It returns True if a model file is found, indicating that the model has been previously saved.
    Otherwise, it returns False.

    Returns:
    - bool: True if a model is saved at the specified path, False otherwise.
    """

    model_file_name = 'random_forest_tennis_model.pkl'
    directory = os.path.abspath('./models')
    model_path = os.path.join(directory, model_file_name)
    return os.path.exists(model_path)


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
    None
    """

    if not model_exists():
        print("Model does not exist, training a new model.")
        data = model_service.load_data()
        print("Preprocess data.")
        df = model_service.preprocess_data(data)
        print("Create label.")
        df = model_service.create_label(df)
        print("Encode categorical features.")
        df = model_service.encode_categorical_features(df)
        print("Handle missing values.")
        df = model_service.handle_missing_values(df)
        print("Train model.")
        classifier = model_service.train_model(df)
        print("Model trained successfully.")
    else:
        print("Model exists, using existing model.")
        model_file_name = 'random_forest_tennis_model.pkl'
        directory = os.path.abspath('./models')
        model_path = os.path.join(directory, model_file_name)
        classifier = joblib.load(model_path)
        print("Model loaded successfully.")

    player1_features = features_service.extract_features(player1)
    player2_features = features_service.extract_features(player2)
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
    features = features_service.merge_match_and_player_features(
        match_features, player1_features, player2_features)

    print("===============================================")
    print("Features for this match", features)
    print("===============================================")

    prediction = model_service.predict_match(classifier, features)
    player1_win_prob = round(prediction[0][1] * 100, 2)
    player2_win_prob = round(prediction[0][0] * 100, 2)

    print("===============================================")
    print("Prediction for the match:")
    print(
        f"{player1} : {player1_win_prob}%")
    print(
        f"{player2} : {player2_win_prob}%")
    print("===============================================")
