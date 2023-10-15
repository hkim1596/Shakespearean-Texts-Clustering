import os
import pandas as pd

# Read the matched_plays.csv
df = pd.read_csv('matched_plays.csv')

# Set the directory path
directory_path = "Data"

# Define a function to create a filename from the author and title
def create_filename(author, title):
    # Extract the first 5 words from the title
    title_words = title.split()[:5]
    
    # Join the author and title words with underscores
    name = "_".join([author] + title_words)
    
    # Remove any characters that are not safe for filenames
    safe_name = "".join([c if c.isalnum() or c in [' ', '_'] else "" for c in name])
    
    # Return the new filename with the .xml extension
    return safe_name + ".xml"

# Loop through each row in the dataframe
for _, row in df.iterrows():
    # Define the current and new file paths
    current_file = os.path.join(directory_path, row['TCP'] + ".xml")
    new_file = os.path.join(directory_path, create_filename(row['Author'], row['Title']))
    
    # Check if the current file exists, then rename it
    if os.path.exists(current_file):
        os.rename(current_file, new_file)

print("Renaming process completed!")
