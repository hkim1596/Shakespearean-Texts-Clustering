import os
import shutil
import pandas as pd

# Load the matched plays CSV
df_matched = pd.read_csv('matched_plays.csv')

# Define the source and target directories
source_dir = "EEBO-TCP"
target_dir = "Data"

# Ensure target directory exists; if not, create it
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Counters for number of TCP values and copied files
num_tcp_values = len(df_matched)
num_copied_files = 0

# Loop through the TCP values and copy matching XML files
for tcp_value in df_matched['TCP']:
    source_file = os.path.join(source_dir, f"{tcp_value}.xml")
    target_file = os.path.join(target_dir, f"{tcp_value}.xml")
    
    if os.path.exists(source_file):
        shutil.copy2(source_file, target_file)
        num_copied_files += 1

# Report the numbers
print(f"Number of TCP plays: {num_tcp_values}")
print(f"Number of copied files: {num_copied_files}")
