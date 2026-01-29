import requests
import json
import re

def check_url(url):
    print(f"Checking {url}...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Origin': 'https://www.coinglass.com',
            'Referer': 'https://www.coinglass.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            if "Extreme Fear" in response.text:
                print("Found 'Extreme Fear' in raw HTML response")
            else:
                print("'Extreme Fear' NOT found in raw HTML response")
                
            # Look for NEXT_DATA
            if '__NEXT_DATA__' in response.text:
                print("Found __NEXT_DATA__")
                match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', response.text, re.DOTALL)
                if match:
                    try:
                        json_content = match.group(1)
                        print("Extracted JSON content length:", len(json_content))
                        json_data = json.loads(json_content)
                    except Exception as e:
                        print(f"JSON parsing failed: {e}")
                else:
                    print("Regex match failed")

    except Exception as e:
        print(f"Error: {e}")

print("--- Scrape Coinglass Page ---")
check_url("https://www.coinglass.com/pro/i/FearGreedIndex")
