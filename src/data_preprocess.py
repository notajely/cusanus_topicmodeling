import os
from bs4 import BeautifulSoup

# Set the current directory (relative to this script)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output folders
xml_folder = os.path.join(current_dir, '../data/raw')
output_folder = os.path.join(current_dir, '../data/processed')

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to process each XML file
def preprocess_tei_data(file_path):
    # Open and read the TEI XML file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the TEI XML structure
    soup = BeautifulSoup(content, 'lxml-xml')

    # Extract the body content (if present)
    body_tag = soup.find('body')
    sermon_text = body_tag.get_text() if body_tag else 'No Content Found'

    # Clean up the text (remove extra whitespace)
    cleaned_text = ' '.join(sermon_text.split())

    # Return the processed text
    return cleaned_text

# Process all XML files in the input folder
for filename in os.listdir(xml_folder):
    if filename.endswith('.xml'):
        file_path = os.path.join(xml_folder, filename)
        processed_sermon = preprocess_tei_data(file_path)

        # Save the processed text to the output folder
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_processed.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(processed_sermon)

print("All files have been successfully preprocessed and saved.")
