import re

with open('debug_page.html', 'r', encoding='utf-8') as f:
    content = f.read()

matches = [m.start() for m in re.finditer('Extreme Fear', content)]
print(f"Found {len(matches)} matches")

for idx in matches:
    print(f"\n--- Match at {idx} ---")
    # escape newlines for printing
    snippet = content[idx-100:idx+100].replace('\n', ' ')
    print(snippet)
