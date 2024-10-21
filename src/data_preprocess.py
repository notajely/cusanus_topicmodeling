import os
from bs4 import BeautifulSoup
from tqdm import tqdm

def preprocess_text(xml_data):
    """
    Process the XML content and extract the word-level text, ensuring to handle line breaks and special characters.
    
    Args:
    xml_data (str): XML content in string format.

    Returns:
    str: Processed text from the XML content.
    """
    try:
        # Parse the XML content using BeautifulSoup
        soup = BeautifulSoup(xml_data, "lxml-xml")
        
        # Extract text within <w> tags
        words = [w.get_text() for w in soup.find_all('w')]
        
        # Join the words and clean up line breaks
        text_output = " ".join(words).replace("\n", " ").strip()

        return text_output
    except Exception as e:
        print(f"Error processing XML: {e}")
        return ""
    
def process_files(input_dir, output_dir):
    """
    Processes all XML files in the input directory and saves the processed text to the output directory.
    
    Args:
    input_dir (str): The directory containing the raw XML files.
    output_dir (str): The directory to save processed text files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    
    for filename in tqdm(files, desc="Processing files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_processed.txt")

        try:
            with open(input_path, 'r', encoding='utf-8') as file:
                xml_content = file.read()

            processed_text = preprocess_text(xml_content)

            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(processed_text)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
    
    print("\nAll files have been successfully processed!")

# Test the preprocess function
def test_preprocess():
    test_xml = '''
    <TEI.2>
      <text>
        <body>
          <p>
            <w id="C160375504" lemma_l="10242">Verbum</w> 
            <w id="C160375505" lemma_l="1433">caro</w> 
            <w id="C160375506" lemma_l="3761">factum</w> 
            <w id="C160375507" lemma_l="9483">est</w>.
          </p>
        </body>
      </text>
    </TEI.2>
    '''

    # Test the preprocess function
    processed_test_text = preprocess_text(test_xml)
    print(f"Processed test text: {processed_test_text}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, '../data/raw/')
    output_dir = os.path.join(current_dir, '../data/processed/')
    
    process_files(input_dir, output_dir)
    
    # test_preprocess()
