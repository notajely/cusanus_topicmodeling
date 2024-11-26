from typing import Dict, Any, List, Tuple
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from scipy.stats import entropy
from itertools import combinations
import logging

def evaluate_single_parameter(params: Dict[str, Any]) -> Dict[str, Any]:
    """评估单个参数组合的性能"""
    try:
        # 提取必要参数
        corpus = params.pop('corpus')
        dictionary = params.pop('dictionary')
        texts = params.pop('texts')
        passes = params.pop('passes', 10)
        random_state = params.pop('random_state', 42)
        
        # 保存词频参数，但不传递给模型
        min_freq = params.pop('min_freq', None)
        max_freq = params.pop('max_freq', None)
        num_topics = params.get('num_topics')
        
        # 训练模型
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            random_state=random_state,
            alpha=params.get('alpha', 'symmetric'),
            eta=params.get('eta', 'symmetric')
        )
        
        # 计算评估指标
        scores = calculate_optimal_score(
            model=model,
            texts=texts,
            dictionary=dictionary
        )
        
        # 获取每个主题的前20个词
        topic_words = []
        for topic_id in range(num_topics):
            words = [word for word, _ in model.show_topic(topic_id, topn=20)]
            topic_words.append(', '.join(words))
        
        # 返回结果时确保所有值都是可序列化的
        return {
            'min_freq': int(min_freq) if min_freq is not None else 2,
            'max_freq': int(max_freq) if max_freq is not None else 200,
            'num_topics': int(num_topics),
            'alpha': str(params.get('alpha', 'symmetric')),
            'eta': str(params.get('eta', 'symmetric')),
            'optimal_score': float(scores['optimal_score']),
            'npmi_score': float(scores['npmi_score']),
            'diversity_score': float(scores['diversity_score']),
            'perplexity': float(scores['perplexity']),
            'model': model,
            'dictionary': dictionary,
            'topic_words': topic_words  # 添加主题词列表
        }
        
    except Exception as e:
        logging.error(f"评估参数时发生错误: {str(e)}")
        return None

def calculate_word_cooccurrences(texts: List[List[str]], window_size: int = 10) -> Dict[Tuple[str, str], int]:
    """计算词对在文档中的共现次数"""
    cooccurrences = {}
    
    for text in texts:
        for i in range(len(text)):
            window = text[max(0, i-window_size):min(len(text), i+window_size+1)]
            for j, word1 in enumerate(window):
                for word2 in window[j+1:]:
                    if word1 < word2:
                        pair = (word1, word2)
                    else:
                        pair = (word2, word1)
                    cooccurrences[pair] = cooccurrences.get(pair, 0) + 1
                    
    return cooccurrences

def calculate_weighted_npmi(model: LdaModel, texts: List[List[str]], 
                          dictionary: Dictionary, top_n: int = 10, 
                          window_size: int = 10, eps: float = 1e-12) -> float:
    """计算加权NPMI分数"""
    try:
        cooccurrences = calculate_word_cooccurrences(texts, window_size)
        
        word_freq = {}
        total_windows = 0
        for text in texts:
            total_windows += max(1, len(text) - window_size + 1)
            for word in text:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        weighted_npmi_sum = 0
        weight_sum = 0
        
        for topic_id in range(model.num_topics):
            topic_words = [word for word, _ in model.show_topic(topic_id, topn=top_n)]
            
            for i, word1 in enumerate(topic_words):
                for word2 in topic_words[i+1:]:
                    if word1 < word2:
                        pair = (word1, word2)
                    else:
                        pair = (word2, word1)
                    
                    count = cooccurrences.get(pair, 0)
                    if count > 0:
                        p_xy = (count + eps) / total_windows
                        p_x = (word_freq.get(word1, 0) + eps) / total_windows
                        p_y = (word_freq.get(word2, 0) + eps) / total_windows
                        
                        pmi = np.log(p_xy / (p_x * p_y))
                        npmi = pmi / (-np.log(p_xy))
                        
                        weighted_npmi_sum += npmi * count
                        weight_sum += count
        
        if weight_sum > 0:
            weighted_npmi = (weighted_npmi_sum / weight_sum + 1) / 2
        else:
            weighted_npmi = 0
            
        return max(0.0, min(1.0, weighted_npmi))
    
    except Exception as e:
        logging.error(f"计算加权NPMI时发生错误: {str(e)}")
        return 0.0

def calculate_improved_diversity(model: LdaModel, beta: float = 0.5, 
                               top_n: int = 10, eps: float = 1e-12) -> float:
    """计算改进的主题多样性分数"""
    try:
        K = model.num_topics
        total_score = 0
        num_pairs = 0
        
        topic_distributions = []
        topic_words = []
        for topic_id in range(K):
            topic_dist = dict(model.show_topic(topic_id, topn=top_n))
            topic_distributions.append([v for _, v in sorted(topic_dist.items())])
            topic_words.append(set(topic_dist.keys()))
        
        for i, j in combinations(range(K), 2):
            M = [(p1 + p2) / 2 for p1, p2 in zip(topic_distributions[i], topic_distributions[j])]
            jsd = (entropy(topic_distributions[i], M) + entropy(topic_distributions[j], M)) / 2
            
            intersection = len(topic_words[i] & topic_words[j])
            min_size = min(len(topic_words[i]), len(topic_words[j]))
            overlap = 1 - (intersection / min_size if min_size > 0 else 0)
            
            pair_score = beta * jsd + (1 - beta) * overlap
            total_score += pair_score
            num_pairs += 1
        
        diversity = total_score / num_pairs if num_pairs > 0 else 0
        return max(0.0, min(1.0, diversity))
    
    except Exception as e:
        logging.error(f"计算改进的多样性分数时发生错误: {str(e)}")
        return 0.0

def calculate_optimal_score(model: LdaModel, texts: List[List[str]], 
                          dictionary: Dictionary, alpha: float = 0.5, 
                          beta: float = 0.5, top_n: int = 10) -> Dict[str, float]:
    """计算最优分数
    
    最优分数 = α * 加权NPMI + (1-α) * 改进的多样性
    
    Args:
        model: LDA模型
        texts: 文档列表
        dictionary: 词典
        alpha: 平衡coherence和diversity的权重 (默认0.5)
        beta: 平衡JSD和词重叠的权重 (默认0.5)
        top_n: 每个主题考虑的前N个词
    
    Returns:
        Dict包含optimal_score, npmi_score, diversity_score和perplexity
    """
    try:
        # 计算加权NPMI
        weighted_npmi = calculate_weighted_npmi(model, texts, dictionary, top_n=top_n)
        
        # 计算改进的多样性
        improved_diversity = calculate_improved_diversity(model, beta=beta, top_n=top_n)
        
        # 计算困惑度
        corpus = [dictionary.doc2bow(doc) for doc in texts]
        perplexity = model.log_perplexity(corpus)
        
        # 计算最终得分
        optimal_score = alpha * weighted_npmi + (1 - alpha) * improved_diversity
        
        return {
            'optimal_score': optimal_score,
            'npmi_score': weighted_npmi,
            'diversity_score': improved_diversity,
            'perplexity': perplexity
        }
    
    except Exception as e:
        logging.error(f"计算最优分数时发生错误: {str(e)}")
        return {
            'optimal_score': 0.0,
            'npmi_score': 0.0,
            'diversity_score': 0.0,
            'perplexity': float('inf')
        }