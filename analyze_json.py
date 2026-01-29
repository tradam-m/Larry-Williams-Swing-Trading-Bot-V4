import json

with open('debug_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def find_path(obj, term, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}"
            if term in k.lower():
                print(f"Found Key '{k}' at {new_path}: {str(v)[:50]}")
            if isinstance(v, (dict, list)):
                find_path(v, term, new_path)
            elif isinstance(v, str) and term in v.lower():
                 print(f"Found Value matching '{term}' in '{k}' at {new_path}: {v[:50]}")

print("--- Searching for 'fear' or 'index' ---")
# find_path(data, "fear")
find_path(data, "index")
