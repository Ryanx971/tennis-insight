from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)


def add_win_rate_columns(df):
    """
    Compute and add win rate, head-to-head win rate, surface win rate, and recent form for each player.
    If no past matches exist for a player, a default value of 0.5 is used.
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

    df['winner_head_to_head_winrate'] = df.apply(lambda row: calculate_head_to_head_win_rate(
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


def calculate_head_to_head_win_rate(match_date, player1_id, player2_id, df):
    """
    Calculate the head-to-head win rate of player1 against player2 for matches played before match_date.

    Returns:
        float: Head-to-head win rate (default 0.5 if no prior matches).
    """
    head_to_head_matches = df[((df['winner_id'] == player1_id) & (df['loser_id'] == player2_id)) | (
        (df['winner_id'] == player2_id) & (df['loser_id'] == player1_id)) & (df['tourney_date'] < match_date)]
    total_head_to_head_matches = len(head_to_head_matches)
    if total_head_to_head_matches > 0:
        player1_wins = len(
            head_to_head_matches[head_to_head_matches['winner_id'] == player1_id])
        return player1_wins / total_head_to_head_matches
    return 0.5


def compute_surface_elo(df):
    """
    Compute surface-specific ELO ratings for each player.
    Sorts the DataFrame chronologically and adds two columns:
    'winner_elo_surface' and 'loser_elo_surface', containing the player's pre-match ELO.

    Returns:
        DataFrame: The DataFrame with added ELO features.
    """
    elo = defaultdict(lambda: defaultdict(lambda: 1500))
    K = 32

    df = df.sort_values(by=["tourney_year", "tourney_month"]).copy()

    for idx, row in df.iterrows():
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

        df.loc[idx, "winner_elo_surface"] = rating_w
        df.loc[idx, "loser_elo_surface"] = rating_l

    return df
