import os
from dotenv import load_dotenv
from services.tennis import analyzer_service


def main():
    load_dotenv()
    analyzer_service.predict(
        os.getenv("FIRST_PLAYER"), os.getenv("SECOND_PLAYER"))


if __name__ == "__main__":
    main()
