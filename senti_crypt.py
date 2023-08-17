import requests
import pandas as pd

def get_senti_crypt():
    """
        This function sends a GET request to the Senticrypt API at 'https://api.senticrypt.com/v2/all.json',
        retrieves the data in JSON format, and converts it into a Pandas DataFrame.

        Returns:
        --------
        pandas.DataFrame or None: If the API request is successful and data is obtained, a Pandas DataFrame
                                   containing the fetched data is returned. If there is an error during the
                                   API request or data retrieval, None is returned, and an error message
                                   is printed to the console.

        """
    url = 'https://api.senticrypt.com/v2/all.json'

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data_list = response.json()
            df = pd.DataFrame(data_list)
            return df
        else:
            print(f"Error: Failed to fetch data. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    data = get_senti_crypt()
    print(data)
