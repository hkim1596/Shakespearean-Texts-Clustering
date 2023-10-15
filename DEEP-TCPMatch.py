import pandas as pd
import os

# Define the path for the Metadata folder
metadata_path = os.path.join(os.getcwd(), 'Metadata')

# Load the datasets from the Metadata folder
deep_csv_path = os.path.join(metadata_path, 'DEEP.csv')
tcp_csv_path = os.path.join(metadata_path, 'TCP.csv')

df1 = pd.read_csv(deep_csv_path)
df2 = pd.read_csv(tcp_csv_path)

# Extract the relevant value from the STC column in df2
df2['extracted_stc'] = df2['STC'].str.extract(r'(?:STC|Wing) ([\dA-Za-z]+)').fillna('')

# Remove leading and trailing whitespaces
df1['stc_or_wing'] = df1['stc_or_wing'].str.strip()
df2['extracted_stc'] = df2['extracted_stc'].str.strip()

# Convert the relevant columns to lowercase for case-insensitive matching
df1['stc_or_wing'] = df1['stc_or_wing'].str.lower()
df2['extracted_stc'] = df2['extracted_stc'].str.lower()

# Rename original "Author" column to "AuthorTCP" in df2
df2.rename(columns={'Author': 'AuthorTCP'}, inplace=True)

# Merge the two dataframes based on the matching columns
merged_df = df1.merge(df2, left_on='stc_or_wing', right_on='extracted_stc', how='inner')

# Add new empty columns at the beginning
for column in ['Exception', 'Type', 'AuthorTitle', 'Author']:
    merged_df.insert(0, column, '')

# Save the matched data to PlayMetadata.csv in the Metadata folder
merged_df.to_csv(os.path.join(metadata_path, 'PlayMetadata.csv'), index=False)

# Save unmatched rows from df1 to a new CSV file in the Metadata folder
unmatched = df1[~df1['stc_or_wing'].isin(df2['extracted_stc'])]
unmatched.to_csv(os.path.join(metadata_path, 'unmatched_plays.csv'), index=False)

# Save a CSV to inspect the values we're trying to match in the Metadata folder
pd.concat([df1['stc_or_wing'], df2['extracted_stc']], axis=1).to_csv(os.path.join(metadata_path, 'stc_values_for_inspection.csv'), index=False)
