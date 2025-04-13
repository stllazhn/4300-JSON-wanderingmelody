import csv
import json

# Define the input and output file paths
csv_file = 'shortened_spotify.csv'  # Change to your CSV file path
json_file = 'shortened_spotify.json'  # Change to your desired JSON output path

# Read the CSV file and convert to JSON
with open(csv_file, mode='r', encoding='utf-8') as infile:
    csv_reader = csv.DictReader(infile)
    rows = list(csv_reader)

# Write the JSON output
with open(json_file, mode='w', encoding='utf-8') as outfile:
    json.dump(rows, outfile, indent=4)

print(f"CSV has been converted to JSON and saved as {json_file}")
