import requests
import time
from typing import Dict, Optional

class CryptoFearAndGreed:
    """
    Retrieves the Crypto Fear and Greed Index.
    
    Primary Source: Alternative.me API
    Note: Coinglass uses the same methodology and data descriptions as Alternative.me 
    (verified by matching 'Surveys' description which notes 'currently paused' on both).
    """
    
    ALTERNATIVE_ME_URL = "https://api.alternative.me/fng/"
    
    @staticmethod
    def get_index() -> Optional[Dict]:
        """
        Fetches the current index value.
        Returns dictionary with:
        - value (int): 0-100
        - classification (str): 'Fear', 'Greed', etc.
        - timestamp (int): Unix timestamp
        - time_until_update (str): Seconds until next update
        """
        try:
            # Usage of limit=1 to get the latest
            response = requests.get(f"{CryptoFearAndGreed.ALTERNATIVE_ME_URL}?limit=1", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                item = data['data'][0]
                return {
                    'value': int(item['value']),
                    'classification': item['value_classification'],
                    'timestamp': int(item['timestamp']),
                    'time_until_update': item.get('time_until_update')
                }
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return None

if __name__ == "__main__":
    print("Fetching Crypto Fear & Greed Index...")
    data = CryptoFearAndGreed.get_index()
    if data:
        print(f"Current Value: {data['value']} ({data['classification']})")
        print(f"Last Update: {time.ctime(data['timestamp'])}")
        if data.get('time_until_update'):
            hours = int(data['time_until_update']) // 3600
            print(f"Next update in approx {hours} hours")
    else:
        print("Failed to retrieve data.")
