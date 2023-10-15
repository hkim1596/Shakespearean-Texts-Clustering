import pandas as pd
import requests
import os
import time

# Ensure the directories exist
if not os.path.exists('Data'):
    os.makedirs('Data')

# Read the matched_plays.csv
df = pd.read_csv('matched_plays.csv')

# Define the base URL and headers
base_url = "https://github.com/textcreationpartnership/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0'
}

# Loop through the TCP values and download the XML files
downloaded_files = 0
for tcp in df['TCP']:
    # Construct the URL
    url = base_url + tcp + "/blob/master/" + tcp + ".xml"
    
    # Fetch the file
    response = requests.get(url, headers=headers)
    
    # If the request was successful, save the XML to the Data folder
    if response.status_code == 200:
        with open(f"Data/{tcp}.xml", "wb") as file:
            file.write(response.content)
        downloaded_files += 1
        print(f"Success to download {tcp}.xml")
    else:
        print(f"Failed to download {tcp}.xml")

    # Add a delay to avoid too many requests in a short period
    time.sleep(5)  # 5-second delay between requests

print(f"Downloaded {downloaded_files} out of {len(df['TCP'])} files.")
