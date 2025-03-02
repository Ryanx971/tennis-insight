import os
import logging
import pandas as pd
import numpy as np
from services.tennis import match_stats_service

logger = logging.getLogger(__name__)


def load_match_data():
    """
    Load ATP tennis match data from multiple CSV files and concatenate them.

    Returns:
        DataFrame: Concatenated match data.
    """
    directory = os.path.abspath('./datasets/tennis/matches')
    files = [
        'atp_matches_2024.csv', 'atp_matches_2023.csv', 'atp_matches_2022.csv',
        'atp_matches_2021.csv', 'atp_matches_2020.csv', 'atp_matches_2019.csv',
        'atp_matches_2018.csv', 'atp_matches_2017.csv', 'atp_matches_2016.csv'
    ]
    data = pd.concat([pd.read_csv(os.path.join(directory, file))
                     for file in files], ignore_index=True)
    selected_levels = ['G', 'M', 'F', 'A']
    data = data[data['tourney_level'].isin(selected_levels)]
    logger.info("Number of rows in match data: %d", data.shape[0])
    return data


def preprocess_match_data(data):
    """
    Preprocess raw match data:
      - Drop unnecessary columns.
      - Remove rows missing critical information.
      - Convert numerical columns.
      - Expand tourney_date into tourney_year and tourney_month.

    Returns:
        DataFrame: Preprocessed match data.
    """
    # Remove the columns that are not useful for our analysis
    df = data.drop(columns=[
        'score', 'winner_name', 'loser_name', 'tourney_name', 'minutes',
        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_ace', 'l_svpt', 'l_SvGms',
        'l_bpFaced', 'l_df', 'l_bpSaved', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'w_SvGms', 'w_ace', 'w_svpt', 'w_bpFaced', 'w_bpSaved', 'w_df',
        'winner_entry', 'loser_entry', 'tourney_id', 'draw_size', 'match_num',
        'winner_hand', 'winner_ht', 'winner_ioc', 'loser_hand', 'loser_ht', 'loser_ioc'
    ])

    # We will delete the entries that do not contain information about important feature
    df.dropna(subset=['winner_rank_points', 'loser_rank_points',
              'winner_rank', 'loser_rank', 'surface'], inplace=True)
    numeric_columns = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age']
    df[numeric_columns] = df[numeric_columns].astype(float)

    # Add winrate columns
    match_stats_service.add_win_rate_columns(df)

    # We will expand the "tourney_date" feature to new columns storing year and month attributes.
    # "tourney_date" is in the format of YYYYMMDD
    df['tourney_year'] = df.tourney_date.astype(str).str[:4].astype(int)
    df['tourney_month'] = df.tourney_date.astype(str).str[4:6].astype(int)
    df = df.drop(columns=['tourney_date'])

    # Add ELO columns
    match_stats_service.compute_surface_elo(df)

    return df


def create_match_labels(df):
    """
    Create labels for match data by duplicating each match:
      - One copy where label = 0 (second player wins).
      - One copy where label = 1 (first player wins).

    Returns:
        DataFrame: Labeled and shuffled match data.
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
    Encode categorical features (surface, tourney_level, round) using LabelBinarizer/LabelEncoder.

    Returns:
        DataFrame: DataFrame with encoded features.
    """
    from sklearn.preprocessing import LabelBinarizer, LabelEncoder
    encoder = LabelBinarizer()
    df['surface'] = encoder.fit_transform(df['surface'].astype(str))
    df['tourney_level'] = LabelEncoder().fit_transform(
        df['tourney_level'].astype(str))
    df['round'] = LabelEncoder().fit_transform(df['round'].astype(str))
    return df


def impute_missing_values(df):
    """
    Handle missing values using SimpleImputer.

    Returns:
        DataFrame: DataFrame with imputed missing values.
    """
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer()
    df_imputed = pd.DataFrame(imputer.fit_transform(df))
    df_imputed.columns = df.columns
    df_imputed.index = df.index
    return df_imputed
