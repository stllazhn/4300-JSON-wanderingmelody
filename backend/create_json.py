import pandas as pd

# Load the CSV file
csv_file = "backend/spotify_millsongdata.csv"  
df = pd.read_csv(csv_file)

# Convert to JSON
json_file = "spotify_millsongdata.json"  
df.to_json(json_file, orient="records", indent=4)

print(f"JSON file saved as {json_file}")