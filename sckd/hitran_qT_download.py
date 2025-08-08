import requests

def download_txt_file(url: str, save_path: str) -> None:
    """
    Downloads a text file from the given URL and saves it to the specified path.

    Parameters:
    - url: str        : The URL of the text file to download.
    - save_path: str  : The local file path where the downloaded file will be saved.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses

        with open(save_path, 'wb') as f:
            f.write(response.content)

        print(f"File successfully downloaded and saved to: {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading the file: {e}")

if __name__ == "__main__":
    for i in range(1, 149):
        url = f"https://hitran.org/data/Q/q{i}.txt"
        save_path = f"data/qT/q{i}.txt"
        download_txt_file(url, save_path)


