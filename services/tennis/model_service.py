import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def add_winrate_columns(df):
    """
    Add winrate columns to the DataFrame.

    Args:
    - df (DataFrame): Input DataFrame containing tennis match data.

    Returns:
    None

    Description:
    This function iterates over each row of the input DataFrame to calculate the winrate of each player based on their past matches. It creates two new columns in the DataFrame, 'winner_winrate' and 'loser_winrate', representing the winrate of the winner and loser of each match, respectively. The winrate is calculated as the percentage of victories out of total matches played for each player up to the current match date. If a player has no past matches, their winrate is set to 50% as a default value.
    """
    start_time = time.time()
    df_copy = df.copy()

    for index, row in df_copy.iterrows():
        current_match_date = row['tourney_date']
        winner_id = row['winner_id']
        loser_id = row['loser_id']

        winner_matches = []
        winner_victories = 0
        loser_matches = []
        loser_victories = 0
        head_to_head_matches = []
        head_to_head_winner_victories = 0

        for index, row in df_copy.iterrows():
            # winner overall winrate
            if (row['tourney_date'] < current_match_date) and (row['winner_id'] == winner_id or row['loser_id'] == winner_id):
                winner_matches.append(row)

            # loser overall winrate
            if (row['tourney_date'] < current_match_date) and (row['winner_id'] == loser_id or row['loser_id'] == loser_id):
                loser_matches.append(row)

            if (row['tourney_date'] < current_match_date) and (row['winner_id'] == loser_id or row['loser_id'] == loser_id):
                loser_matches.append(row)

            # Head-to-head
            if ((row['winner_id'] == winner_id and row['loser_id'] == loser_id) or
                (row['winner_id'] == loser_id and row['loser_id'] == winner_id)) and \
                    row['tourney_date'] < current_match_date:
                head_to_head_matches.append(row)
                if row['winner_id'] == winner_id:
                    head_to_head_winner_victories += 1

        for row in winner_matches:
            if row['winner_id'] == winner_id:
                winner_victories += 1

        for row in loser_matches:
            if row['winner_id'] == loser_id:
                loser_victories += 1

        winner_winrate = 0.5
        if winner_matches:
            winner_winrate = round(
                winner_victories / len(winner_matches), 2)

        loser_winrate = 0.5
        if loser_matches:
            loser_winrate = round(
                loser_victories / len(loser_matches), 2)

        winner_head_to_head_winrate = 0.5
        if head_to_head_matches:
            winner_head_to_head_winrate = round(
                head_to_head_winner_victories / len(head_to_head_matches), 2)

        df['winner_winrate'] = winner_winrate
        df['loser_winrate'] = loser_winrate
        df['winner_head_to_head_winrate'] = winner_head_to_head_winrate
        df['loser_head_to_head_winrate'] = round(
            1 - winner_head_to_head_winrate, 2)

        # print("=====================================================")
        # print("Winner head to head winrate : ", winner_head_to_head_winrate)
        # print("Loser head to head winrate : ", round(
        #     1 - winner_head_to_head_winrate, 2))
        # print("=====================================================")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of add_winrate_columns : {execution_time} seconds")


def load_data():
    """
    Load ATP tennis match data from CSV files.

    Args:
    - directory (str): Directory containing CSV files.

    Returns:
    - df (DataFrame): Concatenated DataFrame of all ATP tennis match data.
    """
    # Five last years of ATP matches are stored in different csv files. We will concatenate them into one single dataframe.
    directory = os.path.abspath('./datasets/tennis/matches')

    files = ['atp_matches_2023.csv', 'atp_matches_2022.csv',
             'atp_matches_2021.csv', 'atp_matches_2020.csv', 'atp_matches_2019.csv', 'atp_matches_2018.csv', 'atp_matches_2017.csv', 'atp_matches_2016.csv']
    data = pd.concat([pd.read_csv(os.path.join(directory, file))
                      for file in files], ignore_index=True)

    selected_levels = ['G', 'M', 'F', 'P', 'PM', 'I']
    data = data[data['tourney_level'].isin(selected_levels)]

    num_rows = data.shape[0]
    print("Number of rows in the DataFrame:", num_rows)

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

    # Add winrate columns
    add_winrate_columns(df)

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
                            "loser_winrate": "first_player_winrate", "loser_head_to_head_winrate": "first_player_head_to_head_winrate",
                            "winner_age": "second_player_age",
                            "winner_rank": "second_player_rank", "winner_rank_points": "second_player_rank_points",
                            "winner_seed": "second_player_seed", "winner_id": "second_player_id", "winner_winrate": "second_player_winrate",
                            "winner_head_to_head_winrate": "second_player_head_to_head_winrate",
                            })
    copy_2_df = df.copy()
    copy_2_df[['first_player_age', 'first_player_rank', 'first_player_rank_points', 'first_player_seed', 'first_player_id', "first_player_winrate", "first_player_head_to_head_winrate",
               'second_player_age', 'second_player_rank', 'second_player_rank_points', 'second_player_seed', 'second_player_id', 'second_player_winrate', "second_player_head_to_head_winrate"]]\
        = copy_2_df[['second_player_age', 'second_player_rank', 'second_player_rank_points', 'second_player_seed', 'second_player_id', 'second_player_winrate', "second_player_head_to_head_winrate",
                    'first_player_age', 'first_player_rank', 'first_player_rank_points', 'first_player_seed', 'first_player_id', 'first_player_winrate', "first_player_head_to_head_winrate"]]

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


def predict_match(classifier, features):
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
            'second_player_winrate': float,
            'first_player_winrate': float,
            'tourney_year': int,
            'tourney_month': int
        }

    Returns:
    - prediction (array): Prediction of the match outcome probability.
    """

    match_data = pd.DataFrame([features])
    return classifier.predict_proba(match_data)
