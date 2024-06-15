import os
import pandas as pd


# Load player data once at service startup
PLAYER_FILE_NAME = 'atp_players.csv'
RANKING_FILE_NAME = 'atp_rankings_current.csv'
PLAYERS_DIRECTORY = os.path.abspath('./datasets/tennis/players')
PLAYERS_DF = pd.read_csv(os.path.join(PLAYERS_DIRECTORY, PLAYER_FILE_NAME))
RANKINGS_DF = pd.read_csv(os.path.join(PLAYERS_DIRECTORY, RANKING_FILE_NAME))


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
    player_name = player.lower()
    player_row = PLAYERS_DF.loc[(PLAYERS_DF['name_first'].str.lower(
    ) + ' ' + PLAYERS_DF['name_last'].str.lower()) == player_name]

    if player_row.empty:
        raise ValueError(f"Player {player_name} not found.")

    player_id = player_row.iloc[0]['player_id']
    ranking_row = RANKINGS_DF.loc[RANKINGS_DF['player'] == player_id]

    if ranking_row.empty:
        raise ValueError(f"Ranking not found for player {player_name}.")

    return {
        'id': player_id,
        # 'age': int(os.getenv("FIRST_PLAYER_AGE")),
        'rank': ranking_row.iloc[0]['rank'],
        'rank_points': ranking_row.iloc[0]['points']
    }


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

    merged_features['first_player_surface_winrate'] = float(
        os.getenv("FIRST_PLAYER_SURFACE_WINRATE"))
    merged_features['second_player_surface_winrate'] = float(
        os.getenv("SECOND_PLAYER_SURFACE_WINRATE"))

    merged_features['first_player_recent_form'] = float(
        os.getenv("FIRST_PLAYER_RECENT_FORM"))
    merged_features['second_player_recent_form'] = float(
        os.getenv("SECOND_PLAYER_RECENT_FORM"))

    # !  Prefer to use the following code to set the age of the players
    merged_features['first_player_age'] = int(os.getenv("FIRST_PLAYER_AGE"))
    merged_features['second_player_age'] = int(os.getenv("SECOND_PLAYER_AGE"))

    desired_order = ['surface', 'tourney_level', 'second_player_id', 'second_player_seed', 'second_player_age', 'first_player_id', 'first_player_seed', 'first_player_age',
                     'best_of', 'round', 'second_player_rank', 'second_player_rank_points', 'first_player_rank', 'first_player_rank_points', 'second_player_winrate',
                     'first_player_winrate', 'second_player_head_to_head_winrate', 'first_player_head_to_head_winrate', 'second_player_surface_winrate', 'first_player_surface_winrate',
                     'second_player_recent_form', 'first_player_recent_form', 'tourney_year', 'tourney_month']

    reordered_features = {key: merged_features[key] for key in desired_order}

    return reordered_features
