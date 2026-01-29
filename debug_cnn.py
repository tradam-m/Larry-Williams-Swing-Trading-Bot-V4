import requests
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Origin': 'https://edition.cnn.com',
    'Referer': 'https://edition.cnn.com/'
}

api_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

print(f"Trying API: {api_url}")
try:
    response = requests.get(api_url, headers=headers, timeout=10)
    print(f"API Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("API Success!")
        
        # Structure often contains 'fear_and_greed' object
        if 'fear_and_greed' in data:
            fg_data = data['fear_and_greed']
            print(f"Score: {fg_data.get('score')}")
            print(f"Rating: {fg_data.get('rating')}")
            print(f"Timestamp: {fg_data.get('timestamp')}")
        else:
            print("Data structure unknown, printing keys:")
            print(data.keys())
            print(str(data)[:200])
    else:
        print("Failed to get data")
        print(response.text[:200])

except Exception as e:
    print(f"Exception: {e}")
