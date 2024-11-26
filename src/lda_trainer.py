from multiprocessing import Pool, cpu_count
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import itertools
import logging
import pandas as pd
from functools import partial
from tqdm import tqdm
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import json

from .evaluator import evaluate_single_parameter
from .data_loader import load_documents, prepare_corpus

class LDATrainer:
    def __init__(self, data_dir: Path, experiment_dir: Optional[Path] = None):
        """Initialize LDA Trainer
        
        Args:
            data_dir: Path to the data directory
            experiment_dir: Directory to save experiment results
            
        Raises:
            FileNotFoundError: If data directory does not exist
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
            
        self.data_dir = data_dir
        self.experiment_dir = experiment_dir or Path('experiments/lda')
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.train_texts: List[Tuple[str, List[str]]] = load_documents(data_dir, 'train_set')
        self.test_texts: List[Tuple[str, List[str]]] = load_documents(data_dir, 'test_set')
        self.dictionary, self.train_corpus = prepare_corpus([doc for _, doc in self.train_texts])
        
    def evaluate_single_parameter(self, params: tuple) -> Dict[str, Any]:
        """Evaluate a single parameter combination
        
        Args:
            params: Tuple of (min_freq, max_freq, num_topics, alpha, eta)
            
        Returns:
            Dictionary containing evaluation results or None if evaluation fails
        """
        min_freq, max_freq, num_topics, alpha, eta = params
        
        try:
            # 1. Create dictionary and corpus from training set
            train_docs = [doc for _, doc in self.train_texts]
            dictionary = Dictionary(train_docs)
            dictionary.filter_extremes(
                no_below=min_freq, 
                no_above=max_freq/len(train_docs)
            )
            train_corpus = [dictionary.doc2bow(doc) for _, doc in self.train_texts]
            
            # 2. Train model on training set
            model = LdaModel(
                corpus=train_corpus,
                id2word=dictionary,
                num_topics=num_topics,
                alpha=alpha,
                eta=eta,
                passes=self.n_passes,
                random_state=42
            )
            
            # 3. Prepare test corpus
            test_corpus = [dictionary.doc2bow(doc) for _, doc in self.test_texts]
            
            # 4. Evaluate on test set
            eval_params = {
                'corpus': test_corpus,
                'dictionary': dictionary,
                'texts': [doc for _, doc in self.test_texts],
                'num_topics': num_topics,
                'alpha': alpha,
                'eta': eta,
                'passes': self.n_passes,
                'random_state': 42,
                'min_freq': min_freq,
                'max_freq': max_freq
            }
            
            # 5. Get performance on test set
            result = evaluate_single_parameter(eval_params)
            if result is None:
                return None
            
            # 6. Add trained model and document topics to result
            result['model'] = model
            result['test_document_topics'] = self.get_document_topics(model, test_corpus)  # 改名为 test_document_topics
            result['test_filenames'] = [doc[0] for doc in self.test_texts]  # 添加测试集文档名
        
            return result
                
        except Exception as e:
            logging.error(f"Failed to evaluate parameter combination: {str(e)}", exc_info=True)
            return None

    def get_document_topics(self, model: LdaModel, corpus: List[List[int]]) -> List[List[Tuple[int, float]]]:
        """Get topic distribution for each document
        
        Args:
            model: Trained LDA model
            corpus: BOW representation of documents
        
        Returns:
            List of topic distributions for each document, where each distribution
            is a list of (topic_id, probability) tuples
            
        Raises:
            ValueError: If model or corpus is invalid
        """
        if not isinstance(model, LdaModel):
            raise ValueError("Invalid LDA model")
        if not corpus:
            raise ValueError("Empty corpus")
            
        return [model.get_document_topics(bow) for bow in corpus]

    def grid_search(self, param_grid: Dict[str, List[Any]], 
                   use_multiprocessing: bool = True,
                   n_passes: int = 100) -> pd.DataFrame:
        """Perform grid search over parameters
        
        Args:
            param_grid: Dictionary of parameter lists to search
            use_multiprocessing: Whether to use multiple processes
            n_passes: Number of passes for LDA training
            
        Returns:
            DataFrame containing evaluation results
            
        Raises:
            ValueError: If parameters are invalid or no valid results
        """
        if not isinstance(param_grid, dict):
            raise ValueError("param_grid must be a dictionary")
        
        if n_passes < 1:
            raise ValueError("n_passes must be positive")
            
        self.n_passes = n_passes
        
        # Generate parameter combinations
        params_combinations = [
            (min_freq, max_freq, num_topic, alpha, eta)
            for min_freq in param_grid.get('min_freqs', [2, 4])
            for max_freq in param_grid.get('max_freqs', [200, 800, 1400, 2000])
            for num_topic in param_grid.get('num_topics', [10])
            for alpha in param_grid.get('alpha_range', ['symmetric', 0.3, 0.5])
            for eta in param_grid.get('eta_range', ['symmetric', 0.3, 0.5])
        ]
        
        try:
            if use_multiprocessing:
                with Pool(processes=cpu_count()) as pool:
                    results = list(tqdm(
                        pool.imap(self.evaluate_single_parameter, params_combinations),
                        total=len(params_combinations)
                    ))
            else:
                results = list(tqdm(
                    map(self.evaluate_single_parameter, params_combinations),
                    total=len(params_combinations)
                ))
            
            # Filter and process results
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                raise ValueError("No valid evaluation results")
                
            results_df = pd.DataFrame(valid_results)
            results_df = results_df.sort_values('optimal_score', ascending=False)
            
            return results_df
            
        except Exception as e:
            logging.error(f"Error during grid search: {str(e)}", exc_info=True)
            raise

    def save_results(self, results_df: pd.DataFrame, prep_type: str) -> Dict[str, Any]:
        """Save evaluation results
        
        Args:
            results_df: DataFrame containing evaluation results
            prep_type: Type of preprocessing used
            
        Returns:
            Dictionary containing detailed results
            
        Raises:
            ValueError: If results_df is invalid
        """
        try:
            # Ensure results_df is a DataFrame
            if not isinstance(results_df, pd.DataFrame):
                raise ValueError("results_df must be a pandas DataFrame")
                
            results_dir = self.experiment_dir / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save basic evaluation results
            results_df.to_csv(results_dir / 'evaluation_results.csv', index=False)
            
            # Save all models and their details
            all_params = []
            for i, row in results_df.iterrows():
                try:
                    model = row.get('model')
                    if not isinstance(model, LdaModel):
                        logging.warning(f"Skipping invalid model at index {i}: {type(model)}")
                        continue
                    
                    # Get topic words
                    topic_words = []
                    for topic_id in range(model.num_topics):
                        words = [word for word, prob in model.show_topic(topic_id, topn=10)]
                        topic_words.append(f"Topic {topic_id+1}: {', '.join(words)}")
                    
                    # Save model
                    model_path = results_dir / f'model_{i+1}.lda'
                    model.save(str(model_path))
                    
                    # Create model parameter dictionary
                    model_info = {
                        'min_freq': int(row['min_freq']),
                        'max_freq': int(row['max_freq']),
                        'alpha': str(row['alpha']),
                        'eta': str(row['eta']),
                        'n_passes': self.n_passes,
                        'test_score': float(row['optimal_score']),
                        'test_npmi': float(row['npmi_score']),
                        'test_diversity': float(row['diversity_score']),
                        'test_perplexity': float(row['perplexity']),
                        'topic_words': '\n'.join(topic_words),
                        'test_document_topics': row.get('test_document_topics', []),  # 保存测试集文档主题分布
                        'test_filenames': row.get('test_filenames', [])  # 保存测试集文档名
                    }
                    
                    all_params.append(model_info)
                    
                except Exception as e:
                    logging.error(f"Error processing model {i+1}: {str(e)}", exc_info=True)
                    continue
            
            # Save detailed results
            detailed_results = {
                'evaluation_metrics': {
                    'optimal_score': float(results_df['optimal_score'].max()),
                    'avg_score': float(results_df['optimal_score'].mean()),
                    'std_score': float(results_df['optimal_score'].std())
                },
                'all_params': all_params
            }
            
            results_path = results_dir / 'detailed_results.json'
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
                
            return detailed_results
            
        except Exception as e:
            logging.error(f"Error saving evaluation results: {str(e)}", exc_info=True)
            raise