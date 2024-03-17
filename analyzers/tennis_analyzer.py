import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    """
    Load ATP tennis match data from CSV files.

    Args:
    - directory (str): Directory containing CSV files.

    Returns:
    - df (DataFrame): Concatenated DataFrame of all ATP tennis match data.
    """
    # Five last years of ATP matches are stored in 5 different csv files. We will concatenate them into one single dataframe.
    directory = os.path.abspath('./datasets/tennis/matches')

    files = ['atp_matches_2023.csv', 'atp_matches_2022.csv',
             'atp_matches_2021.csv', 'atp_matches_2020.csv', 'atp_matches_2019.csv']
    data = pd.concat([pd.read_csv(os.path.join(directory, file))
                      for file in files], ignore_index=True)
    return data


def preprocess_data(data):
    """
    Preprocess ATP tennis match data.

    Args:
    - data (DataFrame): Raw DataFrame of ATP tennis match data.

    Returns:
    - df (DataFrame): Preprocessed DataFrame.
    """
    # Remove the columns that are not useful for our analysis
    df = data.drop(columns=['score', 'winner_name', 'loser_name', 'tourney_name', 'minutes', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_ace', 'l_svpt',
                            'l_SvGms', 'l_bpFaced', 'l_df', 'l_bpSaved',  'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms',
                            'w_ace', 'w_svpt', 'w_bpFaced', 'w_bpSaved', 'w_df', 'winner_entry', 'loser_entry', 'tourney_id',
                            'draw_size', 'match_num', 'winner_hand', 'winner_ht', 'winner_ioc',
                            'loser_hand', 'loser_ht', 'loser_ioc'])

    # We will delete the entries that do not contain information about important feature
    df.dropna(subset=['winner_rank_points', 'loser_rank_points',
                      'winner_rank', 'loser_rank', 'surface'], inplace=True)

    # To avoid analytical errors, we will convert numerical values of string format to float type.
    numeric_columns = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age']
    df[numeric_columns] = df[numeric_columns].astype(float)

    # We will expand the "tourney_date" feature to new columns storing year and month attributes.
    # "tourney_date" is in the format of YYYYMMDD
    df['tourney_year'] = df.tourney_date.astype(str).str[:4].astype(int)
    df['tourney_month'] = df.tourney_date.astype(str).str[4:6].astype(int)
    df = df.drop(columns=['tourney_date'])

    return df


def create_label(df):
    """
    Create labels for ATP tennis match data.

    Args:
    - df (DataFrame): Preprocessed DataFrame.

    Returns:
    - labeled_df (DataFrame): DataFrame with labels.
    """
    # For our supervised prediction model, we have to define our target feature!
    # We will transform our data so that we have 2 players (first player & second player),
    # their respective personal informations (id, hand, age, etc) and general informations about the match and the tourney.
    # Then we will create a column "label" which is equal to 1 if player 1 wins, 0 if player 2 wins.
    # to do so, we will create a first copy of our dataset where the winner is considered as first player so label=0.
    # Then a second copy where we inverse the places of the players so label=1.
    df = df.rename(columns={"loser_age": "first_player_age", "loser_rank": "first_player_rank",
                            "loser_rank_points": "first_player_rank_points",
                            "loser_seed": "first_player_seed", "loser_id": "first_player_id",
                            "winner_age": "second_player_age",
                            "winner_rank": "second_player_rank", "winner_rank_points": "second_player_rank_points",
                            "winner_seed": "second_player_seed", "winner_id": "second_player_id"
                            })
    copy_2_df = df.copy()
    copy_2_df[['first_player_age', 'first_player_rank', 'first_player_rank_points', 'first_player_seed', 'first_player_id',
               'second_player_age', 'second_player_rank', 'second_player_rank_points', 'second_player_seed', 'second_player_id']]\
        = copy_2_df[['second_player_age', 'second_player_rank', 'second_player_rank_points', 'second_player_seed', 'second_player_id',
                    'first_player_age', 'first_player_rank', 'first_player_rank_points', 'first_player_seed', 'first_player_id',]]

    # Second player wins so label=0
    winner_player2 = np.zeros(df.shape[0])
    df['label'] = winner_player2

    # First player wins so label=1
    winner_player1 = np.ones(copy_2_df.shape[0])
    copy_2_df['label'] = winner_player1
    labeled_df = pd.concat([df, copy_2_df])
    labeled_df = labeled_df.sample(frac=1).reset_index(drop=True)
    return labeled_df


def encode_categorical_features(df):
    """
    Encode categorical features in the DataFrame.

    Args:
    - df (DataFrame): DataFrame with categorical features.

    Returns:
    - encoded_df (DataFrame): DataFrame with encoded categorical features.
    """
    encoder = LabelBinarizer()
    df['surface'] = encoder.fit_transform(df['surface'].astype(str))
    df['tourney_level'] = LabelEncoder().fit_transform(
        df['tourney_level'].astype(str))
    df['round'] = LabelEncoder().fit_transform(df['round'].astype(str))
    return df


def handle_missing_values(df):
    """
    Handle missing values in the DataFrame.

    Args:
    - df (DataFrame): DataFrame with missing values.

    Returns:
    - df (DataFrame): DataFrame with missing values handled.
    """
    # Finally, let's handle the few remaing None values using SimpleImputer.
    imputer = SimpleImputer()
    df_imputed = pd.DataFrame(imputer.fit_transform(df))
    df_imputed.columns = df.columns
    df_imputed.index = df.index
    return df_imputed


def train_model(df):
    """
    Train a random forest classifier on the DataFrame.

    Args:
    - df (DataFrame): DataFrame containing features and labels.

    Returns:
    - classifier (RandomForestClassifier): Trained random forest classifier.
    """
    model_file_name = 'random_forest_tennis_model.pkl'
    directory = os.path.abspath('./models')
    model_path = os.path.join(directory, model_file_name)

    if os.path.exists(model_path):
        classifier = joblib.load(model_path)
    else:
        y = df['label']
        X = df.drop(columns='label')
        # Let's now train and execute our prediction model. For this we will use  RandomForest.
        # Split data : 80% for train and 20% for test.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        # Save the new model
        os.makedirs(directory, exist_ok=True)
        joblib.dump(classifier, os.path.join(directory, model_file_name))

        print('Accuracy score : ', accuracy_score(y_test, predictions))

    return classifier


def predict_match(df, classifier, features):
    """
    Predict the outcome of a tennis match between two players.

    Args:
    - df (DataFrame): DataFrame containing features and labels.
    - classifier (RandomForestClassifier): Trained random forest classifier.
    - features (dict): Features of the match.
        {
            'surface': int,
            'tourney_level': int,
            'second_player_id': int,
            'second_player_seed': int,
            'second_player_age': int,
            'first_player_id': int,
            'first_player_seed': int,
            'first_player_age': int,
            'best_of': int,
            'round': int,
            'second_player_rank': int,
            'second_player_rank_points': int,
            'first_player_rank': int,
            'first_player_rank_points': int,
            'tourney_year': int,
            'tourney_month': int
        }

    Returns:
    - prediction (array): Prediction of the match outcome probability.
    """
    match_data = pd.DataFrame([features], columns=df.columns[:-1])
    return classifier.predict_proba(match_data)


def extract_features(player):
    """
    Extract relevant features of the player.

    Args:
    - player (str): player name.

    Returns:
    - player_features (dict): Dictionary containing player features.
        {
            'id': int,
            'age': int,
            'rank': int,
            'rank_points': int
        }
    """
    player_file_name = 'atp_players.csv'
    ranking_file_name = 'atp_rankings_current.csv'
    player_name = player.lower()
    directory = os.path.abspath('./datasets/tennis/players')
    players_df = pd.read_csv(os.path.join(directory, player_file_name))
    players_df['full_name'] = players_df['name_first'].str.lower(
    ) + ' ' + players_df['name_last'].str.lower()
    player_row = players_df[players_df['full_name'] == player_name]

    if not player_row.empty:
        player_id = player_row.iloc[0]['player_id']
        rankings_df = pd.read_csv(os.path.join(directory, ranking_file_name))
        ranking_row = rankings_df[rankings_df['player'] == player_id]

        if not ranking_row.empty:
            return {
                'id': player_id,
                'age': 30,
                'rank': ranking_row.iloc[0]['rank'],
                'rank_points': ranking_row.iloc[0]['points']
            }
        else:
            print(f"Player ID ranking {player_id} not found.")
            sys.exit()
    else:
        print(f"Player {player_name} not found.")
        sys.exit()


def merge_match_and_player_features(match_features, player1_features, player2_features):
    """
    Merge match features with player features of two players.

    Args:
    - match_features (dict): Dictionary containing match features.
    - player1_features (dict): Dictionary containing features of player 1.
    - player2_features (dict): Dictionary containing features of player 2.

    Returns:
    - merged_features (dict): Merged dictionary containing match and player features.
    """
    merged_features = {}
    merged_features.update(match_features)
    merged_features.update(
        {'first_player_' + key: value for key, value in player1_features.items()})
    merged_features.update(
        {'second_player_' + key: value for key, value in player2_features.items()})
    return merged_features


def analyze_tennis_match(player1, player2):
    """
    Analyze a tennis match between two players.

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
    data = load_data()
    df = preprocess_data(data)
    df = create_label(df)
    df = encode_categorical_features(df)
    df = handle_missing_values(df)
    classifier = train_model(df)
    player1_features = extract_features(player1)
    player2_features = extract_features(player2)

    encoder = LabelBinarizer()
    match_features = {
        'surface': encoder.fit_transform(["Hard"])[0],
        'tourney_level': encoder.fit_transform(["M"])[0],
        'best_of': 3,
        'round': encoder.fit_transform(["QF"])[0],
        'tourney_year': 2024,
        'tourney_month': 3,
        'first_player_seed': 32,
        'second_player_seed': 3,
    }
    features = merge_match_and_player_features(
        match_features, player1_features, player2_features)
    prediction = predict_match(df, classifier, features)
    player1_win_prob = round(prediction[0][1] * 100, 2)
    player2_win_prob = round(prediction[0][0] * 100, 2)
    print(
        f"{player1} : {player1_win_prob}%")
    print(
        f"{player2} : {player2_win_prob}%")
