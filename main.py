import os
from dotenv import load_dotenv
from services.tennis import prediction_service


def main():
    load_dotenv()
    first_player = os.getenv("FIRST_PLAYER")
    second_player = os.getenv("SECOND_PLAYER")

    prediction_service.predict_match_outcome(
        first_player, second_player)


if __name__ == "__main__":
    main()
