"""
This module provides functions to preprocess TEI XML files.
"""

# Import necessary modules
import re
import os
import xml.etree.ElementTree as ET
from nltk.corpus import stopwords
from cltk.stem.latin.j_v import JVReplacer
from cltk.tokenize.sentence import TokenizeSentence
from cltk.lemmatize.latin.backoff import BackoffLatinLemmatizer

# Initialize tools
jv_replacer = JVReplacer()
sentence_tokenizer = TokenizeSentence('latin')
lemmatizer = BackoffLatinLemmatizer()
# Load stopwords
stop_words = set(stopwords.words('latin'))

# Define preprocessing function
def preprocess_text(xml_content):
    """
    Function to preprocess TEI XML content for topic modeling.
    Args:
        xml_content (str): Content of the input XML file.
    Returns:
        str: Preprocessed text content.
    """
    # Parse TEI XML
    root = ET.fromstring(xml_content)
    text_elements = root.findall('.//{http://www.tei-c.org/ns/1.0}text')
    text = ""
    for elem in text_elements:
        text += ''.join(elem.itertext())

    # Normalize text
    text = jv_replacer.replace(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()

    # Tokenize sentences and process each sentence
    sentences = sentence_tokenizer.tokenize(text)
    processed_sentences = []
    for sentence in sentences:
        words = sentence.split()
        words = [word for word in words if word not in stop_words]
        lemmatized_words = [lemma for word, lemma in lemmatizer.lemmatize(words)]
        processed_sentence = ' '.join(lemmatized_words)
        processed_sentences.append(processed_sentence)

    # Join processed sentences
    processed_text = ' '.join(processed_sentences)
    return processed_text


# Directory paths
input_dir = '../data/raw'
output_dir = '../data/processed'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process all XML files
for filename in os.listdir(input_dir):
    if filename.endswith('.xml'):
        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'r', encoding='utf-8') as file:
            xml_content = file.read()

        # Preprocess the content
        processed_text = preprocess_text(xml_content)

        # Save to output directory
        output_filename = f"{os.path.splitext(filename)[0]}_preprocessed.txt"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(processed_text)
        print(f'Preprocessed text saved to {output_path}')