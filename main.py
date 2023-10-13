from bs4 import BeautifulSoup

def parse_tei(tei_file):
    """Parse the TEI XML and extract the text content."""
    with open(tei_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'lxml')
        texts = soup.find_all('text')
        # Extract the content from the <text> tag or any specific tag that wraps your content.
        # You might need to adjust the logic based on the structure of your TEI XML files.
        return [text.get_text() for text in texts]

# Assuming you have the TEI files in a directory
import os

tei_directory = 'path_to_tei_files'
all_texts = []

for tei_file in os.listdir(tei_directory):
    if tei_file.endswith('.xml'):
        all_texts.extend(parse_tei(os.path.join(tei_directory, tei_file)))

print(all_texts[:5])  # Print first 5 to check
