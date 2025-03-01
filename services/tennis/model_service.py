from collections import defaultdict
import os
import time
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

    player_ids = set(df['winner_id']) | set(df['loser_id'])
    winrates = {player_id: (0.5, 0.5) for player_id in player_ids}
    surface_winrates = {player_id: {
        surface: 0.5 for surface in df['surface'].unique()} for player_id in player_ids}
    recent_forms = {player_id: 0.5 for player_id in player_ids}

    # Winrate calculation
    for player_id in player_ids:
        player_matches = df[(df['winner_id'] == player_id)
                            | (df['loser_id'] == player_id)]
        total_matches = len(player_matches)
        if total_matches > 0:
            wins = len(
                player_matches[player_matches['winner_id'] == player_id])
            winrate = wins / total_matches
            winrates[player_id] = (winrate, 1 - winrate)

        # Surface winrate calculation
        for surface in df['surface'].unique():
            surface_matches = player_matches[player_matches['surface'] == surface]
            total_surface_matches = len(surface_matches)
            if total_surface_matches > 0:
                surface_wins = len(
                    surface_matches[surface_matches['winner_id'] == player_id])
                surface_winrate = surface_wins / total_surface_matches
                surface_winrates[player_id][surface] = surface_winrate

        # Recent form calculation
        if total_matches >= 10:
            recent_matches = player_matches.head(10)
            recent_wins = len(
                recent_matches[recent_matches['winner_id'] == player_id])
            recent_form = recent_wins / 10
            recent_forms[player_id] = recent_form

    df['winner_winrate'] = df['winner_id'].map(lambda x: winrates[x][0])
    df['loser_winrate'] = df['loser_id'].map(lambda x: winrates[x][0])

    df['winner_head_to_head_winrate'] = df.apply(lambda row: calculate_head_to_head_winrate(
        row['tourney_date'], row['winner_id'], row['loser_id'], df), axis=1)
    df['loser_head_to_head_winrate'] = 1 - df['winner_head_to_head_winrate']

    df['winner_surface_winrate'] = df.apply(
        lambda row: surface_winrates[row['winner_id']][row['surface']], axis=1)
    df['loser_surface_winrate'] = df.apply(
        lambda row: surface_winrates[row['loser_id']][row['surface']], axis=1)

    df['winner_recent_form'] = df['winner_id'].map(lambda x: recent_forms[x])
    df['loser_recent_form'] = df['loser_id'].map(lambda x: recent_forms[x])

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(
        "Execution time of add_winrate_columns : %f seconds", execution_time)


def calculate_head_to_head_winrate(match_date, player1_id, player2_id, df):
    """
    Calculate the head-to-head winrate between two players.

    Args:
    - match_date (int): Date of the match.
    - player1_id (int): ID of the first player.
    - player2_id (int): ID of the second player.
    - df (DataFrame): Input DataFrame containing tennis match data.

    Returns:
    - float: Head-to-head winrate of the first player against the second player.
    """
    head_to_head_matches = df[((df['winner_id'] == player1_id) & (df['loser_id'] == player2_id)) | (
        (df['winner_id'] == player2_id) & (df['loser_id'] == player1_id)) & (df['tourney_date'] < match_date)]
    total_head_to_head_matches = len(head_to_head_matches)
    if total_head_to_head_matches > 0:
        player1_wins = len(
            head_to_head_matches[head_to_head_matches['winner_id'] == player1_id])
        return player1_wins / total_head_to_head_matches
    return 0.5


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

    files = ['atp_matches_2024.csv', 'atp_matches_2023.csv', 'atp_matches_2022.csv',
             'atp_matches_2021.csv', 'atp_matches_2020.csv', 'atp_matches_2019.csv', 'atp_matches_2018.csv', 'atp_matches_2017.csv', 'atp_matches_2016.csv']
    data = pd.concat([pd.read_csv(os.path.join(directory, file))
                      for file in files], ignore_index=True)

    selected_levels = ['G', 'M', 'F', 'P', 'PM', 'I']
    data = data[data['tourney_level'].isin(selected_levels)]

    num_rows = data.shape[0]
    logger.info("Number of rows in the DataFrame: %d", num_rows)

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

    # Add ELO columns
    compute_surface_elo(df)

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
                            "loser_surface_winrate": "first_player_surface_winrate", "loser_recent_form": "first_player_recent_form",
                            "winner_age": "second_player_age",
                            "winner_rank": "second_player_rank", "winner_rank_points": "second_player_rank_points",
                            "winner_seed": "second_player_seed", "winner_id": "second_player_id", "winner_winrate": "second_player_winrate",
                            "winner_head_to_head_winrate": "second_player_head_to_head_winrate",
                            "winner_surface_winrate": "second_player_surface_winrate", "winner_recent_form": "second_player_recent_form"
                            })
    copy_2_df = df.copy()
    copy_2_df[['first_player_age', 'first_player_rank', 'first_player_rank_points', 'first_player_seed', 'first_player_id', "first_player_winrate", "first_player_head_to_head_winrate", 'first_player_surface_winrate', 'first_player_recent_form',
               'second_player_age', 'second_player_rank', 'second_player_rank_points', 'second_player_seed', 'second_player_id', 'second_player_winrate', "second_player_head_to_head_winrate", 'second_player_surface_winrate', 'second_player_recent_form']]\
        = copy_2_df[['second_player_age', 'second_player_rank', 'second_player_rank_points', 'second_player_seed', 'second_player_id', 'second_player_winrate', "second_player_head_to_head_winrate", 'second_player_surface_winrate', 'second_player_recent_form',
                    'first_player_age', 'first_player_rank', 'first_player_rank_points', 'first_player_seed', 'first_player_id', 'first_player_winrate', "first_player_head_to_head_winrate", 'first_player_surface_winrate', 'first_player_recent_form']]

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


def compute_surface_elo(df):
    """
    Sort matches chronologically and compute ELO per surface.
    Store 'winner_elo_surface' / 'loser_elo_surface' in df.
    """
    elo = defaultdict(lambda: defaultdict(lambda: 1500)
                      )  # elo[surface][player_id]
    K = 32

    df = df.sort_values(by=["tourney_year", "tourney_month"]).copy()

    for idx, row in df.iterrows():
        # might be numeric if encoded, or string if not yet encoded
        surface = row["surface"]
        w_id = row["winner_id"]
        l_id = row["loser_id"]

        rating_w = elo[surface][w_id]
        rating_l = elo[surface][l_id]

        expected_w = 1 / (1 + 10 ** ((rating_l - rating_w) / 400))
        expected_l = 1 - expected_w

        new_rating_w = rating_w + K * (1 - expected_w)
        new_rating_l = rating_l + K * (0 - expected_l)

        elo[surface][w_id] = new_rating_w
        elo[surface][l_id] = new_rating_l

        # Store pre-update rating as a feature
        df.loc[idx, "winner_elo_surface"] = rating_w
        df.loc[idx, "loser_elo_surface"] = rating_l

    return df


def evaluate_additional_metrics(model, X_test, y_test):
    """
    Evaluates the model using additional metrics:
    - Log Loss: measures how well the predicted probabilities are calibrated.
    - Brier Score: measures the mean squared difference between predicted probabilities and actual outcomes.
    - Precision, Recall, and F1 Score.
    """
    # Obtain predicted probabilities and predictions
    probs = model.predict_proba(X_test)
    predictions = model.predict(X_test)

    # Compute metrics
    ll = log_loss(y_test, probs)
    bs = brier_score_loss(y_test, probs[:, 1])
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Log results
    logger.info("Log Loss: %.4f", ll)
    logger.info("Brier Score: %.4f", bs)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall: %.4f", recall)
    logger.info("F1 Score: %.4f", f1)

    # Optionally, return these metrics in a dictionary
    return {"log_loss": ll, "brier_score": bs, "precision": precision, "recall": recall, "f1": f1}


def tune_random_forest(X_train, y_train):
    """
    Performs randomized hyperparameter search for a RandomForestClassifier using TimeSeriesSplit.
    Returns the best model found.
    """
    param_dist = {
        "n_estimators": randint(50, 300),
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 10),
    }

    rf_base = RandomForestClassifier(random_state=42)

    # Use TimeSeriesSplit to avoid lookahead bias in time-series data
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
    best_model = random_search.best_estimator_
    return best_model


def train_model(df):
    """
    Trains a RandomForest model using a time-based split (training on data before 2024, testing on 2024),
    performs hyperparameter tuning, logs feature importances, and saves the optimized model.

    Args:
        df (DataFrame): Preprocessed DataFrame containing the 'label' column and date columns 'tourney_year' and 'tourney_month'.

    Returns:
        classifier (RandomForestClassifier): The trained classifier.
    """
    # 1) Sort the DataFrame by date (ascending)
    df = df.sort_values(
        by=["tourney_year", "tourney_month"]).reset_index(drop=True)

    # 2) Time-based split: train on data where tourney_year < 2024, test on data where tourney_year >= 2024
    train_df = df[df["tourney_year"] < 2024]
    test_df = df[df["tourney_year"] >= 2024]

    y_train = train_df["label"]
    X_train = train_df.drop(columns=["label"])
    y_test = test_df["label"]
    X_test = test_df.drop(columns=["label"])

    logger.info("Starting hyperparameter tuning for RandomForest...")
    best_rf = tune_random_forest(X_train, y_train)
    logger.info("Hyperparameter tuning completed.")

    # 3) Evaluate the best model on the 2024 test set
    predictions = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info("Time-based test set accuracy: %.4f", accuracy)

    # After logging accuracy and feature importances
    additional_metrics = evaluate_additional_metrics(best_rf, X_test, y_test)
    logger.info("Additional evaluation metrics: %s", additional_metrics)

    # 4) Log feature importances
    # importances = best_rf.feature_importances_
    # for feature, imp in zip(X_train.columns, importances):
    #     logger.info("Feature '%s' importance: %.4f", feature, imp)

    # 5) Save the optimized model
    model_file_name = 'random_forest_tennis_model.pkl'
    directory = os.path.abspath('./models')
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, model_file_name)
    joblib.dump(best_rf, model_path)
    logger.info("Saved tuned RandomForest model to %s", model_path)

    return best_rf


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
