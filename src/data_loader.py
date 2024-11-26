from pathlib import Path
from typing import List, Tuple, Dict, Optional
from gensim.corpora import Dictionary
import logging
import re

def load_documents(data_dir: Path, subset: str = 'train_set') -> List[Tuple[str, List[str]]]:
    """Load document data
    
    Args:
        data_dir: Path to the data directory
        subset: 'train_set' or 'test_set'
        
    Returns:
        List of tuples, each containing the filename and a list of words in the document
    """
    texts = []
    data_path = data_dir / subset
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    
    try:
        for file_path in sorted(data_path.glob('*.txt')):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                # Split all words directly
                words = content.split()
                
                if words:  # Only add non-empty documents
                    texts.append((file_path.name, words))
                    logging.info(f"Successfully loaded document: {file_path.name}")
                    logging.info(f"Number of words in document: {len(words)}")
                else:
                    logging.warning(f"Skipped empty document: {file_path}")
                    
    except Exception as e:
        logging.error(f"Error loading documents: {str(e)}")
        raise
    
    if not texts:
        raise ValueError(f"No valid documents found in {data_path}")
        
    logging.info(f"Loaded {len(texts)} documents in total")
    return texts

def prepare_corpus(texts: List[List[str]], 
                  min_freq: int = 5, 
                  max_freq: float = 0.5,
                  dictionary: Optional[Dictionary] = None) -> Tuple[Dictionary, List[List[int]]]:
    """Prepare corpus
    
    Args:
        texts: List of documents
        min_freq: Minimum word occurrence
        max_freq: Maximum word occurrence ratio
        dictionary: Optional existing dictionary (for test set)
        
    Returns:
        Tuple of (dictionary, corpus)
    """
    if dictionary is None:
        dictionary = Dictionary(texts)
        # Filter low/high frequency words
        dictionary.filter_extremes(
            no_below=min_freq,
            no_above=max_freq,
            keep_n=None  # Keep all words that meet the criteria
        )
        logging.info(f"Dictionary size: {len(dictionary)}")
    
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Log basic statistics
    total_tokens = sum(len(text) for text in texts)
    unique_tokens = len(set(word for text in texts for word in text))
    avg_doc_length = total_tokens / len(texts)
    
    logging.info(f"Number of documents: {len(texts)}")
    logging.info(f"Total number of words: {total_tokens}")
    logging.info(f"Number of unique words: {unique_tokens}")
    logging.info(f"Average document length: {avg_doc_length:.2f}")
    
    return dictionary, corpus

def plot_word_freq_distribution(texts: List[List[str]], title: str):
    """Plot word frequency distribution"""
    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set font for English
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Count word frequency - 修复: 只使用文档内容(texts中的第二个元素)
    word_freq = Counter([word for _, doc in texts for word in doc])
    words = list(word_freq.keys())
    frequencies = list(word_freq.values())
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # 1. Word frequency-rank distribution (top left)
    plt.subplot(2, 2, 1)
    ranks = range(1, len(frequencies) + 1)
    plt.loglog(ranks, sorted(frequencies, reverse=True), 'b-', alpha=0.7)
    plt.title(f'{title} - Word Frequency-Rank Distribution', fontsize=12)
    plt.xlabel('Word Rank (log scale)', fontsize=10)
    plt.ylabel('Frequency (log scale)', fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 2. Word frequency histogram (top right)
    plt.subplot(2, 2, 2)
    sns.histplot(data=frequencies, bins=50, log_scale=(True, True))
    plt.title(f'{title} - Word Frequency Histogram', fontsize=12)
    plt.xlabel('Frequency (log scale)', fontsize=10)
    plt.ylabel('Number of Words (log scale)', fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 3. Cumulative frequency plot (bottom left)
    plt.subplot(2, 2, 3)
    sorted_freqs = sorted(frequencies, reverse=True)
    cumsum = np.cumsum(sorted_freqs)
    total_words = sum(frequencies)
    plt.plot(range(1, len(cumsum) + 1), cumsum / total_words * 100)
    plt.title(f'{title} - Cumulative Frequency Distribution', fontsize=12)
    plt.xlabel('Number of Words', fontsize=10)
    plt.ylabel('Cumulative Frequency (%)', fontsize=10)
    plt.grid(True)
    
    # 4. Top N frequent words bar chart (bottom right)
    plt.subplot(2, 2, 4)
    top_n = 20
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_words_freq = [freq for word, freq in top_words]
    top_words_labels = [word for word, freq in top_words]
    
    plt.barh(range(top_n), top_words_freq)
    plt.yticks(range(top_n), top_words_labels)
    plt.title(f'{title} - Top {top_n} Frequent Words', fontsize=12)
    plt.xlabel('Frequency', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()

def analyze_datasets():
    """Analyze word frequency distribution of train and test sets"""
    import matplotlib.pyplot as plt
    
    try:
        # Set font for English
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Load data
        base_dir = Path(__file__).resolve().parents[1]
        data_dir = base_dir / 'data'  # Modify data directory path
        
        # Load train and test sets
        train_texts = load_documents(data_dir, 'train_set')
        test_texts = load_documents(data_dir, 'test_set')
        
        # Create directory to save plots
        save_dir = base_dir / 'experiments' / 'word_freq_analysis'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot train set word frequency distribution
        plot_word_freq_distribution(train_texts, 'Train Set')
        plt.savefig(save_dir / 'word_freq_train.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot test set word frequency distribution
        plot_word_freq_distribution(test_texts, 'Test Set')
        plt.savefig(save_dir / 'word_freq_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print and save statistics
        stats = []
        stats.append("\nDataset Statistics:")
        stats.append("-" * 50)
        
        # 修复: 正确访问文档中的词列表（texts中的每个元素是(filename, words)元组）
        train_words = set(word for _, doc in train_texts for word in doc)
        test_words = set(word for _, doc in test_texts for word in doc)
        common_words = train_words & test_words
        
        stats.extend([
            f"Number of documents in train set: {len(train_texts)}",
            f"Number of documents in test set: {len(test_texts)}",
            f"\nVocabulary Statistics:",
            f"Unique words in train set: {len(train_words)}",
            f"Unique words in test set: {len(test_words)}",
            f"Common words: {len(common_words)}",
            f"Words only in train set: {len(train_words - test_words)}",
            f"Words only in test set: {len(test_words - train_words)}",
            f"\nFrequency Statistics:",
            f"Total words in train set: {sum(len(doc) for _, doc in train_texts)}",
            f"Total words in test set: {sum(len(doc) for _, doc in test_texts)}",
            f"Average document length in train set: {sum(len(doc) for _, doc in train_texts) / len(train_texts):.2f}",
            f"Average document length in test set: {sum(len(doc) for _, doc in test_texts) / len(test_texts):.2f}"
        ])
        
        # Print statistics
        print('\n'.join(stats))
        
        # Save statistics to file
        with open(save_dir / 'dataset_statistics.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(stats))
        
    except Exception as e:
        logging.error(f"Error analyzing datasets: {str(e)}")
        raise

if __name__ == "__main__":
    # Set logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run dataset analysis
    analyze_datasets()