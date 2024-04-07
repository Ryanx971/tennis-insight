import os
import sys
import pandas as pd


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
                'age': 20,  # ! This data is gonna be erased later
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

    merged_features['first_player_winrate'] = float(
        os.getenv("FIRST_PLAYER_WINRATE"))
    merged_features['second_player_winrate'] = float(
        os.getenv("SECOND_PLAYER_WINRATE"))
    merged_features['first_player_head_to_head_winrate'] = float(
        os.getenv("FIRST_PLAYER_HEAD_TO_HEAD_WINRATE"))
    merged_features['second_player_head_to_head_winrate'] = float(
        os.getenv("SECOND_PLAYER_HEAD_TO_HEAD_WINRATE"))

    # !  Prefer to use the following code to set the age of the players
    merged_features['first_player_age'] = int(os.getenv("FIRST_PLAYER_AGE"))
    merged_features['second_player_age'] = int(os.getenv("SECOND_PLAYER_AGE"))

    desired_order = ['surface', 'tourney_level', 'second_player_id', 'second_player_seed', 'second_player_age', 'first_player_id', 'first_player_seed', 'first_player_age',
                     'best_of', 'round', 'second_player_rank', 'second_player_rank_points', 'first_player_rank', 'first_player_rank_points', 'second_player_winrate', 'first_player_winrate', 'second_player_head_to_head_winrate', 'first_player_head_to_head_winrate', 'tourney_year', 'tourney_month']

    reordered_features = {key: merged_features[key] for key in desired_order}

    return reordered_features
