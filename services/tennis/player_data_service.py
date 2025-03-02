import os
import pandas as pd

# File names and directory for player data
PLAYER_FILE_NAME = 'atp_players.csv'
RANKING_FILE_NAME = 'atp_rankings_current.csv'
PLAYERS_DIRECTORY = os.path.abspath('./datasets/tennis/players')

# Load CSV files once at startup
PLAYERS_DF = pd.read_csv(os.path.join(PLAYERS_DIRECTORY, PLAYER_FILE_NAME))
RANKINGS_DF = pd.read_csv(os.path.join(PLAYERS_DIRECTORY, RANKING_FILE_NAME))


def get_players_dataframe() -> pd.DataFrame:
    """
    Returns the players DataFrame loaded from the CSV.
    """
    return PLAYERS_DF


def get_rankings_dataframe() -> pd.DataFrame:
    """
    Returns the rankings DataFrame loaded from the CSV.
    """
    return RANKINGS_DF
