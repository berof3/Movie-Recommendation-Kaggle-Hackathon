import kagglehub
import shutil
import os

def fetch_competition_data():
    """ This function Ingest the data Automatically:
    Downloads the dataset via API and moves it to the project's data/raw folder."""

    # Definning the project destination
    destination_path = "data/raw"

    # Skip re-downloading/copying if the files are already present locally
    if os.path.isdir(destination_path) and os.listdir(destination_path):
        print(f"Data already present in {destination_path}, skipping download.")
        return

    print("Starting data ingestion via API")

    # Downloading the latest version of the competition files
    download_path = kagglehub.competition_download("movie-recommendation-hackathon-2026")

    # Moving the files from cache to our local "data/raw" folder
    print(f"Moving files to {destination_path}...")
    shutil.copytree(download_path, destination_path, dirs_exist_ok=True)

    print("\n Data Ingestion Completed")

if __name__=="__main__":
    fetch_competition_data()
